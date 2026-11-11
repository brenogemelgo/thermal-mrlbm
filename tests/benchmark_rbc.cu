#include "../src/initialConditions.cuh"
#include "../src/kernel.cuh"
#include "../src/rbcStatistics.cuh"
#include <algorithm>

// Short, deterministic coupled-solver benchmark. RBC_BENCH_SPLIT_NEE is used
// only when compiling against an untouched copy of the original implementation.
static void advance(real_t *&moments, real_t *&buffer, natural_t step)
{
    streamCollide<<<dim3(GRID_X, GRID_Y, GRID_Z), dim3(BLOCK_NX, BLOCK_NY, BLOCK_NZ)>>>(moments, buffer, step);
#if defined(RBC_BENCH_SPLIT_NEE)
    applyNEEPlates<<<dim3((NX + 31) / 32, (NY + 3) / 4, 2), dim3(32, 4)>>>(buffer, step + 1);
#endif
    std::swap(moments, buffer);
}

template <typename F>
static double milliseconds(F &&work)
{
    cudaEvent_t begin, end;
    outputCheckCuda(cudaEventCreate(&begin), "create benchmark event");
    outputCheckCuda(cudaEventCreate(&end), "create benchmark event");
    outputCheckCuda(cudaEventRecord(begin), "record benchmark start");
    work();
    outputCheckCuda(cudaEventRecord(end), "record benchmark end");
    outputCheckCuda(cudaEventSynchronize(end), "finish benchmark work");
    float elapsed = 0;
    outputCheckCuda(cudaEventElapsedTime(&elapsed, begin, end), "read benchmark event");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return elapsed;
}

template <typename Kernel>
static void resources(const char *name, Kernel kernel, int threads)
{
    cudaFuncAttributes attributes;
    int blocks = 0;
    cudaDeviceProp device;
    outputCheckCuda(cudaGetDeviceProperties(&device, 0), "query benchmark device");
    outputCheckCuda(cudaFuncGetAttributes(&attributes, kernel), "query kernel resources");
    outputCheckCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, 0), "query kernel occupancy");
    std::cout << name << " registers=" << attributes.numRegs
              << " local_bytes=" << attributes.localSizeBytes
              << " shared_bytes=" << attributes.sharedSizeBytes
              << " resident_blocks=" << blocks
              << " theoretical_occupancy=" << double(blocks * threads) / device.maxThreadsPerMultiProcessor << '\n';
}

int main(int argc, char **argv)
{
    try
    {
        const natural_t steps = argc > 1 ? std::stoul(argv[1]) : 200;
        if (!steps)
            throw std::runtime_error("benchmark steps must be positive");

        setSimulationRunId("benchmark");
        createOutputDirectories();
        size_t freeBefore = 0, total = 0, freeAfter = 0;
        outputCheckCuda(cudaMemGetInfo(&freeBefore, &total), "query free memory");
        real_t *moments = nullptr, *buffer = nullptr;
        const size_t bytes = size_t(CELLS) * NUM_MOMENTS * sizeof(real_t);
        outputCheckCuda(cudaMalloc(&moments, bytes), "allocate benchmark state");
        outputCheckCuda(cudaMalloc(&buffer, bytes), "allocate benchmark buffer");
        rbc::Statistics statistics;
        statistics.initializeSeries(false, 0);
        initializeCase<<<dim3(GRID_X, GRID_Y, GRID_Z), dim3(BLOCK_NX, BLOCK_NY, BLOCK_NZ)>>>(moments, buffer);
        applyNEEPlates<<<dim3((NX + 31) / 32, (NY + 3) / 4, 2), dim3(32, 4)>>>(moments);
        outputCheckCuda(cudaDeviceSynchronize(), "initialize benchmark");
        outputCheckCuda(cudaMemGetInfo(&freeAfter, &total), "query allocated memory");

        resources("streamCollide", streamCollide, BLOCK_NX * BLOCK_NY * BLOCK_NZ);
        resources("reduceTiles", rbc::reduceTiles, rbc::THREADS);
        resources("reducePlanes", rbc::reducePlanes, rbc::THREADS);
        std::cout << "state_and_reduction_bytes=" << 2 * bytes + (size_t(rbc::TILES) + 1) * rbc::PROFILE_SIZE * sizeof(double)
                  << " cuda_memory_delta_bytes=" << freeBefore - freeAfter << '\n';

        for (natural_t step = 0; step < 20; ++step)
            advance(moments, buffer, step);
        outputCheckCuda(cudaDeviceSynchronize(), "warm up benchmark");
        std::vector<double> times;
        for (int repeat = 0; repeat < 5; ++repeat)
            times.push_back(milliseconds([&]
            {
                for (natural_t step = 0; step < steps; ++step)
                    advance(moments, buffer, step + 20);
            }));
        std::sort(times.begin(), times.end());
        std::cout << "coupled_gpu_ms_per_step=" << times[2] / steps
                  << " coupled_gpu_mlups=" << double(CELLS) * steps / (times[2] * 1000) << '\n';

        const double tiles = milliseconds([&]
        {
            for (int repeat = 0; repeat < 20; ++repeat)
                rbc::reduceTiles<<<dim3(rbc::TILES, NZ), rbc::THREADS>>>(moments, statistics.partial);
        }) / 20;
        const double planes = milliseconds([&]
        {
            for (int repeat = 0; repeat < 20; ++repeat)
                rbc::reducePlanes<<<rbc::PLANE_BLOCKS, rbc::THREADS>>>(statistics.partial, statistics.deviceProfile);
        }) / 20;
        outputCheckCuda(cudaGetLastError(), "benchmark reductions");
        std::cout << "tiles_ms_per_sample=" << tiles << " planes_ms_per_sample=" << planes << '\n';

        const auto start = std::chrono::steady_clock::now();
        for (natural_t step = 1; step <= steps; ++step)
        {
            advance(moments, buffer, step - 1);
            if (step % ActiveCase::timeAverageStride == 0)
                statistics.sample(moments, step);
        }
        outputCheckCuda(cudaDeviceSynchronize(), "finish benchmark");
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << "sample_interval=" << ActiveCase::timeAverageStride
                  << " end_to_end_s=" << elapsed
                  << " end_to_end_mlups=" << double(CELLS) * steps / (elapsed * 1.e6) << '\n';
        cudaFree(moments);
        cudaFree(buffer);
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
