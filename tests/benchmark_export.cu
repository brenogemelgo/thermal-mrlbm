#include "../src/output.cuh"

__global__ void seedExport(real_t *fields, unsigned revision = 0)
{
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= CELLS) return;
    uint32_t bits = uint32_t(i) * 747796405u + 2891336453u;
    bits = ((bits >> ((bits >> 28) + 4)) ^ bits) * 277803737u;
    const real_t perturbation = real_t((bits >> 22) ^ bits) / real_t(UINT32_MAX);
    fields[i] = real_t(1) - real_t(i / STRIDE) / real_t(NZ - 1) + real_t(.05) * (perturbation - real_t(.5)) + real_t(.001) * revision;
    for (size_t d = 1; d < 4; ++d) fields[d * CELLS + i] = real_t(.05) * (perturbation - real_t(.5)) / real_t(d);
}

__global__ void packExport(const real_t *fields, real_t *packed, natural_t z, unsigned count)
{
    const natural_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < STRIDE)
        for (unsigned f = 0; f < count; ++f)
            packed[size_t(f) * STRIDE + i] = fields[size_t(f) * CELLS + size_t(z) * STRIDE + i];
}

static void original(const real_t *fields, unsigned components, natural_t step)
{
    const auto start = snapshot::Clock::now();
    const auto path = getSimulationHdf5Directory() / "fields" / (outputStepName(step) + ".h5");
    auto temporary = path;temporary += ".tmp";
    std::filesystem::create_directories(path.parent_path());
    real_t *packed = nullptr;
    moment_io::check(cudaMalloc(&packed, size_t(components) * STRIDE * sizeof(real_t)), "benchmark packing allocation");
    std::vector<real_t> host(size_t(components) * STRIDE);
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);cudaEventCreate(&end);
    double packing = 0, transfer = 0, writing = 0, flushing = 0;
    {
        h5out::Handle file(H5Fcreate(temporary.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
        for (auto pair : {std::pair{"step", std::to_string(step)}, {"run_id", simulationRunId()},
                          {"velocity_convention", std::string("physical_u")}, {"axis_order", std::string("zyx")},
                          {"lattice_spacing", std::string("1")}, {"wall_nodes", std::string("0,NZ-1")}})
            h5out::attribute(file, pair.first, pair.second);
        h5out::Handle group(h5out::group(file, "fields"), H5Gclose);
        const char *names[] = {"temperature", "ux", "uy", "uz"};
        std::vector<std::unique_ptr<h5out::Handle>> data;
        for (unsigned f = 0; f < components; ++f)
            data.emplace_back(new h5out::Handle(h5out::dataset(group, names[f], sizeof(real_t) == 4 ? H5T_IEEE_F32LE : H5T_IEEE_F64LE,
                {NZ, NY, NX}, {1, std::min(64u, NY), std::min(128u, NX)}), H5Dclose));
        for (natural_t z = 0; z < NZ; ++z)
        {
            cudaEventRecord(begin);
            packExport<<<(STRIDE + 255) / 256, 256>>>(fields, packed, z, components);
            cudaEventRecord(end);cudaEventSynchronize(end);
            float milliseconds = 0;cudaEventElapsedTime(&milliseconds, begin, end);
            packing += milliseconds / 1000.;
            const auto copy = snapshot::Clock::now();
            moment_io::check(cudaMemcpy(host.data(), packed, host.size() * sizeof(real_t), cudaMemcpyDeviceToHost), "benchmark pageable transfer");
            transfer += snapshot::seconds(copy);
            const auto write = snapshot::Clock::now();
            for (unsigned f = 0; f < components; ++f)
                h5out::slab(*data[f], sizeof(real_t) == 4 ? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE,
                            host.data() + size_t(f) * STRIDE, {z, 0, 0}, {1, NY, NX});
            writing += snapshot::seconds(write);
        }
        const auto flush = snapshot::Clock::now();
        h5out::flush(file);flushing = snapshot::seconds(flush);
    }
    std::filesystem::rename(temporary, path);
    cudaEventDestroy(begin);cudaEventDestroy(end);cudaFree(packed);
    std::cout << "original components=" << components << " packing_gpu_s=" << packing << " d2h_s=" << transfer
              << " hdf5_write_filter_s=" << writing << " flush_s=" << flushing
              << " total_s=" << snapshot::seconds(start) << " file_bytes=" << std::filesystem::file_size(path) << '\n';
}

int main(int argc, char **argv)
{
    try
    {
        if (argc == 4 && std::string(argv[1]) == "--snapshot-worker")
            return snapshot::worker(3, 4, std::stoull(argv[2]), std::stol(argv[3]));
        if (argc < 5) throw std::runtime_error("usage: benchmark_export ROOT original4|original1|sync|async SNAPSHOTS COMPRESSION [WORKERS]");
        simulationOutputRoot() = argv[1];setSimulationRunId(argv[2]);
        snapshot::mode() = argv[2];h5out::compression() = std::stoi(argv[4]);
        if (argc > 5) snapshot::workers() = std::stoi(argv[5]);
        if (argc > 6) snapshot::chunkRows() = std::stoi(argv[6]);
        if (argc > 7) snapshot::chunkColumns() = std::stoi(argv[7]);
        const int snapshots = std::stoi(argv[3]);
        createOutputDirectories();
        real_t *fields = nullptr;
        moment_io::check(cudaMalloc(&fields, size_t(4) * CELLS * sizeof(real_t)), "benchmark fields allocation");
        seedExport<<<(CELLS + 255) / 256, 256>>>(fields);
        moment_io::check(cudaDeviceSynchronize(), "seed benchmark fields");
        const auto start = snapshot::Clock::now();
        snapshot::TemperatureOutput output;
        for (int i = 0; i < snapshots; ++i)
        {
            if (i) seedExport<<<(CELLS + 255) / 256, 256>>>(fields, i);
            const auto capture = snapshot::Clock::now();
            if (snapshot::mode() == "original4" || snapshot::mode() == "original1")
                original(fields, snapshot::mode() == "original4" ? 4 : 1, i);
            else output.save(fields, i, true);
            std::cout << "submission step=" << i << " seconds=" << snapshot::seconds(capture) << '\n';
        }
        output.drain();
        std::cout << "completed total_s=" << snapshot::seconds(start) << '\n';
        cudaFree(fields);
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';return 1;
    }
}
