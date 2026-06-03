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

#if defined(CASE_CAVITY)
    std::cout << "RA: " << static_cast<double>(ActiveCase::RA) << std::endl;
    std::cout << "GBETA: " << static_cast<double>(ActiveCase::GBETA) << std::endl;
#elif defined(CASE_JET)
    std::cout << "RI: " << static_cast<double>(ActiveCase::RI) << std::endl;
    std::cout << "GBETA: " << static_cast<double>(ActiveCase::GBETA) << std::endl;
#endif
}

int main(int argc, char **argv)
{
    bool continueFromCheckpoint = false;

    for (int arg = 1; arg < argc; ++arg)
    {
        const std::string argument(argv[arg]);

        if (argument == "--continue" || argument == "continue")
        {
            continueFromCheckpoint = true;
        }
        else
        {
            std::cerr << "Unknown argument: " << argument << std::endl;
            return EXIT_FAILURE;
        }
    }

    initializeOutputLayout();

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

    CUDA_CHECK(cudaFree(momentsAlloc));
    CUDA_CHECK(cudaFree(dbufferAlloc));
    return 0;
}
