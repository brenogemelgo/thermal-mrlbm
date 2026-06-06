#include "boundary/hostInitialization.cuh"
#include "initialConditions.cuh"
#include "kernel.cuh"
#include "output.cuh"

// #define BENCHMARK

#define CUDA_CHECK(call)                                                         \
    do                                                                           \
    {                                                                            \
        const cudaError_t err = (call);                                          \
        if (err != cudaSuccess)                                                  \
        {                                                                        \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (false)

static void printCaseSummary()
{
    std::cout << "selected case: " << ActiveCase::NAME << std::endl;
    std::cout << "grid: " << NX << " x " << NY << " x " << NZ << std::endl;
    std::cout << "NU: " << static_cast<double>(VISCOSITY) << std::endl;
    std::cout << "ALPHA: " << static_cast<double>(THERMAL_DIFFUSIVITY) << std::endl;
    std::cout << "RE: " << static_cast<double>(REYNOLDS) << std::endl;
    std::cout << "PR: " << static_cast<double>(PRANDTL) << std::endl;

#if defined(CASE_RAYLEIGH_BENARD)
    std::cout << "RA: " << static_cast<double>(ActiveCase::RA) << std::endl;
    std::cout << "GBETA: " << static_cast<double>(ActiveCase::GBETA) << std::endl;
    std::cout << "H: " << static_cast<double>(ActiveCase::H) << std::endl;
    std::cout << "U_FREEFALL: " << static_cast<double>(ActiveCase::U_FREEFALL) << std::endl;
#elif defined(CASE_JET)
    std::cout << "RI: " << static_cast<double>(ActiveCase::RI) << std::endl;
    std::cout << "GBETA: " << static_cast<double>(ActiveCase::GBETA) << std::endl;
#endif
}

static bool isValidRunId(
    const std::string &runId)
{
    return !runId.empty() &&
           runId != "." &&
           runId != ".." &&
           runId.find('/') == std::string::npos &&
           runId.find('\\') == std::string::npos;
}

int main(int argc, char **argv)
{
    bool continueFromCheckpoint = false;
    std::string runId("000");

    for (int arg = 1; arg < argc; ++arg)
    {
        const std::string argument(argv[arg]);

        if (argument == "--continue" || argument == "continue")
        {
            continueFromCheckpoint = true;
        }
        else if (argument == "--runId")
        {
            if (arg + 1 >= argc)
            {
                std::cerr << "Missing value for --runId" << std::endl;
                return EXIT_FAILURE;
            }
            runId = argv[++arg];
        }
        else if (argument.rfind("--runId=", 0) == 0)
        {
            runId = argument.substr(std::strlen("--runId="));
        }
        else if (!argument.empty() && argument[0] != '-' && runId == "000")
        {
            runId = argument;
        }
        else
        {
            std::cerr << "Unknown argument: " << argument << std::endl;
            return EXIT_FAILURE;
        }
    }

    if (!isValidRunId(runId))
    {
        std::cerr << "Invalid run id: " << runId << std::endl;
        return EXIT_FAILURE;
    }

    setSimulationRunId(runId);
    initializeOutputLayout();

    real_t *moments = nullptr;
    real_t *dbuffer = nullptr;
    constexpr size_t bytes = static_cast<size_t>(NUM_MOMENTS) * static_cast<size_t>(CELLS) * sizeof(real_t);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&moments), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dbuffer), bytes));

    real_t *momentsAlloc = moments;
    real_t *dbufferAlloc = dbuffer;

#if defined(CASE_JET)
    constexpr size_t scalarBytes = static_cast<size_t>(CELLS) * sizeof(real_t);
    real_t *temperatureAccum = nullptr;
    real_t *uzAccum = nullptr;
    real_t *temperatureMean = nullptr;
    real_t *uzMean = nullptr;
    natural_t timeAverageCount = 0u;

    if constexpr (ActiveCase::timeAverageEnabled)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&temperatureAccum), scalarBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&uzAccum), scalarBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&temperatureMean), scalarBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&uzMean), scalarBytes));
        CUDA_CHECK(cudaMemset(temperatureAccum, 0, scalarBytes));
        CUDA_CHECK(cudaMemset(uzAccum, 0, scalarBytes));
    }
#endif

    constexpr dim3 block(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
    constexpr dim3 grid(GRID_X, GRID_Y, GRID_Z);

    CUDA_CHECK(cudaFuncSetCacheConfig(streamCollide, cudaFuncCachePreferL1));
    CUDA_CHECK(initIRBCBoundaryTables());

    natural_t startStep = 0;
    if (continueFromCheckpoint)
    {
        startStep = loadLatestCheckpoint(moments, dbuffer);
    }
    else
    {
        initializeCase<<<grid, block>>>(moments, dbuffer);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::cout << std::endl;
    if (continueFromCheckpoint)
    {
        std::cout << "simulation continue from step " << startStep << std::endl;
    }
    else
    {
        std::cout << "simulation start" << std::endl;
    }
    std::cout << "output: " << getSimulationOutputDirectory() << std::endl;
    std::cout << "binaries: " << getBinaryOutputDirectory() << std::endl;
    std::cout << "vtis: " << getVtiOutputDirectory() << std::endl;
    printCaseSummary();

#ifndef BENCHMARK
    writeOutput(moments, startStep);
#endif

    const auto start = std::chrono::high_resolution_clock::now();
#ifndef BENCHMARK
    auto lastStamp = start;
    natural_t lastStampStep = startStep;
#endif

    for (natural_t t = startStep; t < NSTEPS; ++t)
    {
        streamCollide<<<grid, block>>>(moments, dbuffer, t);
        std::swap(moments, dbuffer);

#if defined(CASE_JET)
        if constexpr (ActiveCase::timeAverageEnabled)
        {
            constexpr natural_t timeAverageStride = ActiveCase::timeAverageStride > 0u ? ActiveCase::timeAverageStride : 1u;
            const natural_t sampleStep = t + 1u;
            if (sampleStep >= ActiveCase::timeAverageStart &&
                ((sampleStep - ActiveCase::timeAverageStart) % timeAverageStride) == 0u)
            {
                accumulateJetTimeAverage<<<grid, block>>>(moments, temperatureAccum, uzAccum);
                ++timeAverageCount;
            }
        }
#endif

#ifndef BENCHMARK
        if ((t + 1) % STAMP == 0)
        {
            CUDA_CHECK(cudaDeviceSynchronize());

            const auto now = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> stampElapsed = now - lastStamp;

            const natural_t stampSteps = (t + 1) - lastStampStep;
            const double stampMlups = static_cast<double>(CELLS) * static_cast<double>(stampSteps) / stampElapsed.count() / static_cast<double>(1000000);

            std::cout << std::endl;
            std::cout << "step " << (t + 1) << " / " << NSTEPS << std::endl;
            std::cout << "MLUPS: " << stampMlups << std::endl;

            writeOutput(moments, t + 1);

            lastStamp = std::chrono::high_resolution_clock::now();
            lastStampStep = t + 1;
        }
#endif
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    const natural_t completedSteps = NSTEPS > startStep ? NSTEPS - startStep : 0;
    const double mlups = static_cast<double>(CELLS) * static_cast<double>(completedSteps) / elapsed.count() / static_cast<double>(1000000);

    std::cout << std::endl;
    std::cout << "elapsed: " << elapsed.count() << " s" << std::endl;
    std::cout << "MLUPS: " << mlups << std::endl;

#if defined(CASE_JET)
    if constexpr (ActiveCase::timeAverageEnabled)
    {
        if (timeAverageCount > 0u)
        {
            finalizeJetTimeAverage<<<grid, block>>>(
                temperatureAccum,
                uzAccum,
                temperatureMean,
                uzMean,
                timeAverageCount);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            writeScalarOutput(temperatureMean, "temperatureMean");
            writeScalarOutput(uzMean, "uzMean");
        }

        writeMetadata(timeAverageCount);
        std::cout << "time-average samples: " << timeAverageCount << std::endl;
    }
#endif

    CUDA_CHECK(cudaFree(momentsAlloc));
    CUDA_CHECK(cudaFree(dbufferAlloc));
#if defined(CASE_JET)
    if constexpr (ActiveCase::timeAverageEnabled)
    {
        CUDA_CHECK(cudaFree(temperatureAccum));
        CUDA_CHECK(cudaFree(uzAccum));
        CUDA_CHECK(cudaFree(temperatureMean));
        CUDA_CHECK(cudaFree(uzMean));
    }
#endif
    return 0;
}
