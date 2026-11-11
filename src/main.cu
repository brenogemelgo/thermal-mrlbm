#if defined(CASE_JET)
#include "boundary/hostInitialization.cuh"
#endif
#include "initialConditions.cuh"
#include "kernel.cuh"
#include "checkpoint.cuh"
#include "validation/output.cuh"
#if defined(CASE_RAYLEIGH_BENARD)
#include "multiGpu/runtime.cuh"
#endif

#define CUDA_CHECK(call) outputCheckCuda((call), #call)

static bool validRunId(const std::string &s)
{
    return !s.empty() && s != "." && s != ".." && s.find('/') == s.npos && s.find('\\') == s.npos;
}
static natural_t parseStep(const std::string &s)
{
    size_t end = 0;
    const auto n = std::stoull(s, &end);
    if (end != s.size() || n > UINT32_MAX || s.empty() || s[0] == '-')
        throw std::runtime_error("invalid step count");
    return static_cast<natural_t>(n);
}
static int run(int argc, char **argv)
{
    std::cout.setf(std::ios::unitbuf);
    bool resume = false, fields = false;
    natural_t gpuCount = 1;
    std::array<int, 2> gpuDevices = {0, 1};
    std::string haloTransport = "peer";
    bool emulateTwoGpus = false;
    bool gpuInfo = false;
    std::string runId = "000";
    natural_t endStep = NSTEPS, checkpointEvery = STAMP;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--continue" || arg == "continue")
            resume = true;
        else if (arg == "--gpu-info")
            gpuInfo = true;
        else if (arg == "--vti" || arg == "--vti-every")
            throw std::runtime_error("runtime VTI export is disabled; use convertVTI <HDF5 location> offline");
#if defined(MRL_MULTI_GPU_TESTING)
        else if (arg == "--emulate-two-gpus")
        {
            emulateTwoGpus = true;
            gpuCount = 2;
            gpuDevices = {0, 0};
        }
#endif
        else if (arg == "--fields")
            fields = true;
        else if (arg == "--slice-velocity")
            outputSchedule().sliceVelocity = true;
        else if (arg == "--runId" || arg == "--steps" || arg == "--checkpoint-every" || arg == "--output-root" || arg == "--hdf5-root" ||
                 arg == "--slice-every" || arg == "--field-every" || arg == "--profile-every" ||
                 arg == "--progress-every" || arg == "--output-start" || arg == "--keep-checkpoints" || arg == "--compression" ||
                 arg == "--gpus" || arg == "--devices" || arg == "--halo-transport" ||
                 arg == "--field-io" || arg == "--io-workers")
        {
            if (++i >= argc)
                throw std::runtime_error("missing value for " + arg);
            if (arg == "--field-io")
                snapshot::mode() = argv[i];
            else if (arg == "--io-workers")
                snapshot::workers() = parseStep(argv[i]);
            else if (arg == "--gpus")
                gpuCount = parseStep(argv[i]);
            else if (arg == "--halo-transport")
                haloTransport = argv[i];
            else if (arg == "--devices")
            {
                const std::string devices = argv[i];
                const auto comma = devices.find(',');
                gpuDevices[0] = parseStep(devices.substr(0, comma));
                gpuDevices[1] = comma == devices.npos ? -1 : int(parseStep(devices.substr(comma + 1)));
            }
            else if (arg == "--runId")
                runId = argv[i];
            else if (arg == "--output-root")
                simulationOutputRoot() = argv[i];
            else if (arg == "--hdf5-root")
                simulationHdf5Root() = argv[i];
            else if (arg == "--steps")
                endStep = parseStep(argv[i]);
            else if (arg == "--checkpoint-every")
                checkpointEvery = parseStep(argv[i]);
            else if (arg == "--slice-every")
                outputSchedule().slices = parseStep(argv[i]);
            else if (arg == "--field-every")
                outputSchedule().fields = parseStep(argv[i]);
            else if (arg == "--profile-every")
                outputSchedule().profiles = parseStep(argv[i]);
            else if (arg == "--progress-every")
                outputSchedule().progress = parseStep(argv[i]);
            else if (arg == "--output-start")
                outputSchedule().start = parseStep(argv[i]);
            else if (arg == "--keep-checkpoints")
                outputSchedule().keepCheckpoints = parseStep(argv[i]);
            else
                h5out::compression() = parseStep(argv[i]);
        }
        else if (arg.rfind("--runId=", 0) == 0)
            runId = arg.substr(8);
        else if (!arg.empty() && arg[0] != '-' && runId == "000")
            runId = arg;
        else
            throw std::runtime_error("unknown argument: " + arg);
    }
    if (gpuInfo)
    {
#if defined(CASE_RAYLEIGH_BENARD)
        two_gpu::printInfo();
        return 0;
#else
        throw std::runtime_error("--gpu-info is provided by RBC builds");
#endif
    }
    if (!validRunId(runId) || checkpointEvery == 0 || h5out::compression() > 9 || outputSchedule().keepCheckpoints == 1)
        throw std::runtime_error("invalid run ID, interval, compression (0..9), or retention (0=all, >=2)");
    if (gpuCount != 1 && gpuCount != 2)
        throw std::runtime_error("--gpus must be 1 or 2");
    if ((snapshot::mode() != "async" && snapshot::mode() != "sync" && snapshot::mode() != "serial") ||
        snapshot::workers() == 0 || snapshot::workers() > 32)
        throw std::runtime_error("field IO must be async, sync or serial; io-workers must be 1..32");
#if !defined(CASE_RAYLEIGH_BENARD)
    if (gpuCount != 1)
        throw std::runtime_error("two-GPU decomposition is specialized to RBC");
#endif
    if (fields && !outputSchedule().fields)
        outputSchedule().fields = checkpointEvery;
    outputSchedule().checkpoints = checkpointEvery;
    outputSchedule().end = endStep;
    setSimulationRunId(runId);
    createOutputDirectories();
    if (!resume && std::filesystem::exists(statisticsPath()))
        throw std::runtime_error("existing statistics; use --continue or another run ID");
#if defined(CASE_RAYLEIGH_BENARD) || defined(CASE_JET)
    if (!resume && std::filesystem::exists(checkpointDirectory()) && !std::filesystem::is_empty(checkpointDirectory()))
        throw std::runtime_error("run already has checkpoints; choose another run ID or use --continue");
#endif
    real_t *moments = nullptr, *buffer = nullptr;
    constexpr size_t bytes = size_t(NUM_MOMENTS) * CELLS * sizeof(real_t);
#if defined(CASE_RAYLEIGH_BENARD)
    std::unique_ptr<two_gpu::Solver> parallel;
    if (gpuCount == 2)
    {
        parallel = std::make_unique<two_gpu::Solver>(gpuDevices, haloTransport, emulateTwoGpus);
        moments = parallel->current();
        buffer = parallel->previous();
    }
    else
#endif
    {
        CUDA_CHECK(cudaSetDevice(gpuDevices[0]));
        CUDA_CHECK(cudaMalloc(&moments, bytes));
        CUDA_CHECK(cudaMalloc(&buffer, bytes));
    }
    auto synchronizeSolver = [&]
    {
#if defined(CASE_RAYLEIGH_BENARD)
        if (parallel)
        {
            parallel->synchronize();
            return;
        }
#endif
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    constexpr dim3 grid(GRID_X, GRID_Y, GRID_Z), block(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
    natural_t startStep = 0;
#if defined(CASE_RAYLEIGH_BENARD)
    rbc::Statistics stats(!parallel);
    auto sampleStatistics = [&](natural_t step)
    {
        if (parallel)
            parallel->sample(stats, step);
        else
            stats.sample(moments, step);
    };
    if (resume)
    {
        startStep = restoreCheckpoint(moments, stats);
        if (parallel)
            parallel->prime();
    }
#else
    CUDA_CHECK(initIRBCBoundaryTables());
    // Preserve the jet's full-field averages; RBC uses only reduced FP64 profiles.
    double *ta = nullptr, *ua = nullptr;
    real_t *tm = nullptr, *um = nullptr;
    uint64_t count = 0;
    CUDA_CHECK(cudaMalloc(&ta, size_t(CELLS) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ua, size_t(CELLS) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&tm, size_t(CELLS) * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&um, size_t(CELLS) * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(ta, 0, size_t(CELLS) * sizeof(double)));
    CUDA_CHECK(cudaMemset(ua, 0, size_t(CELLS) * sizeof(double)));
    if (resume)
        startStep = restoreJetCheckpoint(moments, count, ta, ua);
#endif
    if (!resume)
    {
#if defined(CASE_RAYLEIGH_BENARD)
        if (parallel)
            parallel->initialize();
        else
#endif
        {
            initializeCase<<<grid, block>>>(moments, buffer);
            CUDA_CHECK(cudaGetLastError());
#if defined(CASE_RAYLEIGH_BENARD)
            applyNEEPlates<<<dim3((NX + 31) / 32, (NY + 3) / 4, 2), dim3(32, 4)>>>(moments);
#endif
        }
    }
    if (endStep < startStep)
        throw std::runtime_error("requested end step precedes checkpoint");
    // The checkpoint is authoritative. Replay output newer than it on resume.
    if (resume)
        for (const auto &folder : {"fields", "slices", "validation"})
        {
            const auto dir = getSimulationHdf5Directory() / folder;
            if (!std::filesystem::exists(dir))
                continue;
            for (const auto &entry : std::filesystem::directory_iterator(dir))
            {
                const auto stem = entry.path().stem().string();
                const auto prefix = std::string(folder) == "validation" ? "snapshot_" : "step_";
                if (entry.path().extension() == ".h5" && stem.rfind(prefix, 0) == 0 &&
                    std::stoull(stem.substr(std::strlen(prefix))) > startStep)
                    std::filesystem::remove(entry.path());
            }
        }
    writeMetadata();
#if defined(CASE_RAYLEIGH_BENARD)
    stats.initializeSeries(resume, startStep);
    if (resume)
        stats.exportAverages();
    if (!resume)
        sampleStatistics(0);
#endif
#if defined(RBC_VALIDATION_CASE)
    validation::SliceOutput validationOutput;
    if (!resume)
        validationOutput.save(moments, moments, 0);
#endif
    synchronizeSolver();
    FieldOutput fieldOutput;
    if (!resume && outputSchedule().start == 0)
    {
        if (outputSchedule().slices)
            fieldOutput.save(moments, 0, true);
        if (outputSchedule().fields)
            fieldOutput.save(moments, 0, false);
        fieldOutput.drain();
    }
    std::cout << "case " << ActiveCase::NAME << " grid " << NX << 'x' << NY << 'x' << NZ << " RR" << RR_ORDER
              << " nu=" << VISCOSITY << " kappa=" << THERMAL_DIFFUSIVITY << " tau=" << TAU_SHEAR
              << " tau_T=" << TAU_THERMAL << " steps " << startStep << ".." << endStep << '\n';
    std::cout << "step " << startStep << " / " << endStep << " (" << (endStep ? 100.0 * startStep / endStep : 100.0)
              << "%) elapsed=0s window=NA MLUPS cumulative=NA MLUPS ETA=NA (new timing session)\n";
    const auto begin = std::chrono::steady_clock::now();
    auto previous = begin;
    natural_t previousStep = startStep;
    const double initialIO = h5out::ioSeconds();
    const double initialWriter = fieldOutput.backgroundSeconds();
    double previousIO = initialIO;
    auto due = [](natural_t step, natural_t interval)
    { return interval && step % interval == 0; };
    for (natural_t step = startStep + 1; step <= endStep; ++step)
    {
        bool saved = false;
#if defined(CASE_RAYLEIGH_BENARD)
        if (parallel)
        {
            parallel->advance(step - 1);
            moments = parallel->current();
            buffer = parallel->previous();
        }
        else
#endif
        {
            streamCollide<<<grid, block>>>(moments, buffer, step - 1);
            std::swap(moments, buffer);
            CUDA_CHECK(cudaGetLastError());
        }
#if defined(RBC_VALIDATION_CASE)
        if (step % ActiveCase::timeAverageStride == 0 || step == endStep)
        {
            synchronizeSolver();
            h5out::OutputTimer timer;
            validationOutput.save(moments, buffer, step);
            saved = true;
        }
#endif
#if defined(CASE_RAYLEIGH_BENARD)
        // Fixed lattice-step sampling; first average is strictly after the transient.
        if ((step <= ActiveCase::timeAverageStart && step % ActiveCase::timeAverageStride == 0) ||
            (step > ActiveCase::timeAverageStart && (step - ActiveCase::timeAverageStart) % ActiveCase::timeAverageStride == 0))
        {
            sampleStatistics(step);
            saved = true;
        }
#else
        if (ActiveCase::timeAverageEnabled && step > ActiveCase::timeAverageStart &&
            (step - ActiveCase::timeAverageStart) % ActiveCase::timeAverageStride == 0)
        {
            accumulateThermalTimeAverage<<<(CELLS + 255) / 256, 256>>>(moments, ta, ua);
            ++count;
        }
#endif
        if (step >= outputSchedule().start && (due(step, outputSchedule().slices) || due(step, outputSchedule().fields) ||
                                               (step == endStep && (outputSchedule().slices || outputSchedule().fields))))
        {
            h5out::OutputTimer timer;
            if (due(step, outputSchedule().slices) || (step == endStep && outputSchedule().slices) || snapshot::mode() == "serial")
            {
                h5out::ComputeWaitTimer computation;
                synchronizeSolver();
            }
            if (due(step, outputSchedule().slices) || (step == endStep && outputSchedule().slices))
                fieldOutput.save(moments, step, true);
            if (due(step, outputSchedule().fields) || (step == endStep && outputSchedule().fields))
                fieldOutput.save(moments, step, false);
            saved = true;
        }
#if defined(CASE_RAYLEIGH_BENARD)
        if (due(step, outputSchedule().profiles) || step == endStep)
        {
            synchronizeSolver();
            h5out::OutputTimer timer;
            stats.exportAverages();
            writeMetadata(stats.count);
            saved = true;
        }
#endif
        if (due(step, checkpointEvery) || step == endStep)
        {
            synchronizeSolver();
            h5out::OutputTimer timer;
            // Commit snapshots through this step before their checkpoint/history.
            fieldOutput.drain();
#if defined(CASE_RAYLEIGH_BENARD)
            saveCheckpoint(moments, step, stats);
            stats.exportAverages();
            writeMetadata(stats.count);
#else
            saveJetCheckpoint(moments, step, count, ta, ua);
#endif
            if (outputSchedule().keepCheckpoints)
            {
                std::vector<std::filesystem::path> paths;
                for (const auto &entry : std::filesystem::directory_iterator(checkpointDirectory()))
                    if (entry.path().extension() == ".chk")
                        paths.push_back(entry.path());
                std::sort(paths.begin(), paths.end());
                while (paths.size() > outputSchedule().keepCheckpoints)
                {
                    std::filesystem::remove(paths.front());
                    std::filesystem::remove(checkpointHistoryDirectory() / ("statistics_" + paths.front().stem().string() + ".h5"));
                    paths.erase(paths.begin());
                }
            }
            saved = true;
        }
        const bool progressDue = due(step, outputSchedule().progress) || step == endStep;
        if (saved || progressDue)
        {
            fieldOutput.poll();
        }
        if (progressDue)
        {
            // Saves above already wait for their device work, including the
            // statistics profile copy. Only a heartbeat needs another wait.
            if (!saved)
            {
                synchronizeSolver();
            }
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - begin).count();
            const double window = std::chrono::duration<double>(now - previous).count();
            const double io = h5out::ioSeconds() - initialIO, windowIO = h5out::ioSeconds() - previousIO;
            const double updates = double(CELLS) * (step - startStep) / 1.e6;
            const double windowRate = double(CELLS) * (step - previousStep) / 1.e6 / window;
            const double rate = updates / elapsed, compute = std::max(0.0, elapsed - io);
            const double eta = elapsed * (endStep - step) / double(step - startStep);
            {
                h5out::OutputTimer timer;
                std::cout << std::fixed << std::setprecision(3) << "step " << step << " / " << endStep
                          << " (" << 100.0 * step / endStep << "%) elapsed=" << elapsed << "s window=" << windowRate
                          << " MLUPS cumulative=" << rate << " MLUPS compute=" << compute << "s output=" << io
                          << "s compute_MLUPS=" << (compute > 0 ? updates / compute : 0)
                          << " writer=" << fieldOutput.backgroundSeconds() - initialWriter << "s ETA=" << eta << "s\n";
                h5out::Handle file(h5out::file(statisticsPath()), H5Fclose);
                h5out::Handle group(h5out::group(file, "statistics"), H5Gclose);
                h5out::append(group, "runtime", "step,session_start,percent,elapsed_s,window_s,window_mlups,cumulative_mlups,compute_s,output_s,window_output_s,compute_mlups,eta_s",
                              {double(step), double(startStep), 100.0 * step / endStep, elapsed, window, windowRate, rate, compute, io, windowIO, compute > 0 ? updates / compute : 0, eta});
                h5out::flush(file);
            }
            previous = now;
            previousStep = step;
            previousIO = initialIO + io;
        }
    }
    {
        h5out::OutputTimer timer;
        fieldOutput.drain();
    }
    synchronizeSolver();
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    std::cout << "elapsed " << elapsed << " s; " << double(CELLS) * (endStep - startStep) / elapsed / 1.e6 << " MLUPS\n";
#if defined(CASE_RAYLEIGH_BENARD)
    if (endStep == startStep)
    {
        stats.exportAverages();
        writeMetadata(stats.count);
    }
    std::cout << "averaged samples " << stats.count << '\n';
#else
    if (count)
    {
        finalizeThermalTimeAverage<<<(CELLS + 255) / 256, 256>>>(ta, ua, tm, um, count);
        CUDA_CHECK(cudaGetLastError());
        writeScalarOutput(tm, "temperatureMean");
        writeScalarOutput(um, "uzMean");
    }
    writeMetadata(count);
    cudaFree(ta);
    cudaFree(ua);
    cudaFree(tm);
    cudaFree(um);
#endif
#if defined(CASE_RAYLEIGH_BENARD)
    if (parallel)
        parallel.reset();
    else
#endif
    {
        CUDA_CHECK(cudaFree(moments));
        CUDA_CHECK(cudaFree(buffer));
    }
    return 0;
}
int main(int argc, char **argv)
{
    try
    {
        if (argc == 4 && std::string(argv[1]) == "--snapshot-worker")
            return snapshot::worker(3, 4, std::stoull(argv[2]), std::stol(argv[3]));
        return run(argc, argv);
    }
    catch (const std::exception &e)
    {
        std::cerr << "thermal-mrlbm: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
