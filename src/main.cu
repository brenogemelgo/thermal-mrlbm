#include "boundary/hostInitialization.cuh"
#include "initialConditions.cuh"
#include "kernel.cuh"
#include "output.cuh"

// #define STATISTICS
// #define BENCHMARK

#ifdef STATISTICS
#include "statistics.cuh"
#endif

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

static bool startsWith(
    const std::string &value,
    const char *prefix)
{
    const std::size_t prefixLength = std::strlen(prefix);
    return value.size() >= prefixLength && value.compare(0, prefixLength, prefix) == 0;
}

static bool isValidCaseId(
    const std::string &caseId)
{
    return !caseId.empty() &&
           caseId != "." &&
           caseId != ".." &&
           caseId.find('/') == std::string::npos &&
           caseId.find('\\') == std::string::npos;
}

int main(int argc, char **argv)
{
    bool continueFromCheckpoint = false;
    bool inletTurbulenceDiagnostics = false;
    std::string caseId = "default";

    for (int arg = 1; arg < argc; ++arg)
    {
        const std::string argument(argv[arg]);

        if (argument == "--continue" || argument == "continue")
        {
            continueFromCheckpoint = true;
        }
        else if (argument == "--inlet-turbulence-diagnostics")
        {
            inletTurbulenceDiagnostics = true;
        }
        else if (argument == "--case-id")
        {
            if (arg + 1 >= argc)
            {
                std::cerr << "Missing value for --case-id" << std::endl;
                return EXIT_FAILURE;
            }
            caseId = argv[++arg];
        }
        else if (startsWith(argument, "--case-id="))
        {
            caseId = argument.substr(std::strlen("--case-id="));
        }
        else if (!argument.empty() && argument[0] != '-' && caseId == "default")
        {
            caseId = argument;
        }
        else
        {
            std::cerr << "Unknown argument: " << argument << std::endl;
            return EXIT_FAILURE;
        }
    }

    if (!isValidCaseId(caseId))
    {
        std::cerr << "Invalid case id: " << caseId << std::endl;
        return EXIT_FAILURE;
    }

    if (inletTurbulenceDiagnostics)
    {
        printSyntheticTurbulentInflowDiagnostics();
        return EXIT_SUCCESS;
    }

    setSimulationOutputDirectory(std::filesystem::path("output") / caseId);

    real_t *moments = nullptr;
    real_t *dbuffer = nullptr;
    constexpr size_t bytes = static_cast<size_t>(NUM_MOMENTS) * static_cast<size_t>(CELLS) * sizeof(real_t);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&moments), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dbuffer), bytes));

    real_t *momentsAlloc = moments;
    real_t *dbufferAlloc = dbuffer;

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
        jetInit<<<grid, block>>>(moments, dbuffer);
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
    std::cout << "case id: " << caseId << std::endl;
    std::cout << "output: " << getSimulationOutputDirectory() << std::endl;

#ifndef BENCHMARK
    writeOutput(moments, startStep);
#endif

#ifdef STATISTICS
    JetStatisticsAccumulator statisticsAccumulator;
    natural_t statisticsSampleCount = 0;
    accumulateStatisticsSample(statisticsAccumulator, moments);
    ++statisticsSampleCount;
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

#ifdef STATISTICS
        accumulateStatisticsSample(statisticsAccumulator, moments);
        ++statisticsSampleCount;
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

#ifdef STATISTICS
    writeStatisticsOutput(statisticsAccumulator, statisticsSampleCount, startStep, NSTEPS);
#endif

    CUDA_CHECK(cudaFree(momentsAlloc));
    CUDA_CHECK(cudaFree(dbufferAlloc));
    return 0;
}
