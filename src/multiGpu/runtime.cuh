#pragma once

#include "kernels.cuh"
#include <array>
#include <memory>

namespace two_gpu
{
inline void printInfo()
{
    int count = 0;
    moment_io::check(cudaGetDeviceCount(&count), "enumerate GPUs");
    std::cout << "{\"devices\":[";
    for (int device = 0; device < count; ++device)
    {
        cudaDeviceProp properties;
        moment_io::check(cudaGetDeviceProperties(&properties, device), "query GPU");
        if (device)
        {
            std::cout << ',';
        }
        std::cout << "{\"index\":" << device << ",\"name\":" << std::quoted(properties.name)
                  << ",\"total_memory_bytes\":" << properties.totalGlobalMem << '}';
    }
    std::cout << "],\"z_split\":" << SPLIT_Z << ",\"peer_access\":[";
    for (int source = 0; source < count; ++source)
    {
        if (source)
        {
            std::cout << ',';
        }
        std::cout << '[';
        for (int destination = 0; destination < count; ++destination)
        {
            int access = source == destination;
            if (source != destination)
            {
                moment_io::check(cudaDeviceCanAccessPeer(&access, source, destination), "query peer capability");
            }
            if (destination)
            {
                std::cout << ',';
            }
            std::cout << access;
        }
        std::cout << ']';
    }
    std::cout << "]}\n";
}

class Solver final : public moment_io::Distributed
{
    enum class Transport {Peer, Host, Local};
    struct Part
    {
        int device = -1;
        natural_t begin = 0, planes = 0, cells = 0;
        std::string name;
        real_t *state[2] = {}, *send[2] = {}, *halo[2] = {}, *pinned[2] = {};
        double *partial = nullptr, *profile = nullptr;
        real_t *output = nullptr;
        std::vector<real_t> outputHost;
#if defined(RBC_VALIDATION_CASE)
        double *validation = nullptr;
        std::vector<double> validationHost;
#endif
        cudaStream_t bulk = nullptr, skin = nullptr, copy = nullptr, upload = nullptr;
        cudaStream_t snapshotCopy = nullptr;
        cudaEvent_t snapshotBegin = nullptr, snapshotEnd = nullptr;
        cudaEvent_t bulkDone[2] = {}, skinDone[2] = {}, downloaded[2] = {}, transferred[2] = {};
        bool enabledPeer = false;
    };

    std::array<Part, 2> parts;
    Transport transport = Transport::Peer;
    int parity = 0;
    bool emulated = false;
    double *hostProfiles = nullptr;
    static constexpr size_t haloBytes = size_t(NUM_MOMENTS) * STRIDE * sizeof(real_t);

    static void check(cudaError_t error, const char *operation)
    {
        moment_io::check(error, operation);
    }

    void primary() const
    {
        check(cudaSetDevice(parts[0].device), "select primary GPU");
    }

    int buffer(const real_t *token) const
    {
        if (token == parts[0].state[0])
        {
            return 0;
        }
        if (token == parts[0].state[1])
        {
            return 1;
        }
        throw std::runtime_error("unrecognized distributed moment buffer");
    }

    template <typename F>
    static void ranks(F &&function)
    {
        // Unlike the device constexpr_for helper, this host dispatch must allow
        // allocation/launch errors to propagate to the owner's cleanup handler.
        function(IntegralConstant<natural_t, 0>{});
        function(IntegralConstant<natural_t, 1>{});
    }

    // Each transfer is one contiguous 14-field face. The destination's previous
    // skin must finish reading this halo before the buffer is overwritten.
    void exchangeOne(int rank, int next, int previous, bool selected = false)
    {
        Part &source = parts[rank];
        Part &destination = parts[1 - rank];
        if (!selected)
        {
            check(cudaSetDevice(source.device), "select halo source");
        }
        check(cudaStreamWaitEvent(source.copy, source.skinDone[next], 0), "wait packed interface");
        if (transport == Transport::Host)
        {
            check(cudaMemcpyAsync(source.pinned[next], source.send[next], haloBytes, cudaMemcpyDeviceToHost, source.copy), "stage outgoing halo");
            check(cudaEventRecord(source.downloaded[next], source.copy), "record staged halo");
            check(cudaSetDevice(destination.device), "select halo destination");
            check(cudaStreamWaitEvent(destination.upload, source.downloaded[next], 0), "wait staged halo");
            check(cudaStreamWaitEvent(destination.upload, destination.skinDone[previous], 0), "protect incoming halo reuse");
            check(cudaMemcpyAsync(destination.halo[next], source.pinned[next], haloBytes, cudaMemcpyHostToDevice, destination.upload), "upload incoming halo");
            check(cudaEventRecord(source.transferred[next], destination.upload), "record incoming halo");
        }
        else
        {
            check(cudaStreamWaitEvent(source.copy, destination.skinDone[previous], 0), "protect incoming halo reuse");
            if (transport == Transport::Peer)
            {
                check(cudaMemcpyPeerAsync(
                    destination.halo[next], destination.device,
                    source.send[next], source.device, haloBytes, source.copy), "exchange peer halo");
            }
            else
            {
                check(cudaMemcpyAsync(destination.halo[next], source.send[next], haloBytes, cudaMemcpyDeviceToDevice, source.copy), "exchange test halo");
            }
            check(cudaEventRecord(source.transferred[next], source.copy), "record peer halo");
        }
        if (transport == Transport::Host)
        {
            check(cudaSetDevice(source.device), "restore source GPU after staging");
        }
    }

    void exchange(int next, int previous)
    {
        for (int rank = 0; rank < 2; ++rank)
        {
            exchangeOne(rank, next, previous);
        }
        primary();
    }

    void release() noexcept
    {
        if (moment_io::distributed() == this) moment_io::distributed() = nullptr;
        for (auto &part : parts)
        {
            if (part.device < 0) continue;
            cudaSetDevice(part.device);
            cudaDeviceSynchronize();
        }
        for (int rank = 0; rank < 2; ++rank)
        {
            auto &part = parts[rank];
            if (part.device < 0) continue;
            cudaSetDevice(part.device);
            for (int i = 0; i < 2; ++i)
            {
                if (part.state[i]) cudaFree(part.state[i]);
                if (part.send[i]) cudaFree(part.send[i]);
                if (part.halo[i]) cudaFree(part.halo[i]);
                if (part.pinned[i]) cudaFreeHost(part.pinned[i]);
                if (part.bulkDone[i]) cudaEventDestroy(part.bulkDone[i]);
                if (part.skinDone[i]) cudaEventDestroy(part.skinDone[i]);
                if (part.downloaded[i]) cudaEventDestroy(part.downloaded[i]);
            }
            if (part.partial) cudaFree(part.partial);
            if (part.profile) cudaFree(part.profile);
            if (part.output) cudaFree(part.output);
#if defined(RBC_VALIDATION_CASE)
            if (part.validation) cudaFree(part.validation);
#endif
            if (part.bulk) cudaStreamDestroy(part.bulk);
            if (part.skin) cudaStreamDestroy(part.skin);
            if (part.copy) cudaStreamDestroy(part.copy);
            if (part.upload) cudaStreamDestroy(part.upload);
            if (part.snapshotCopy) cudaStreamDestroy(part.snapshotCopy);
            if (part.snapshotBegin) cudaEventDestroy(part.snapshotBegin);
            if (part.snapshotEnd) cudaEventDestroy(part.snapshotEnd);
            if (part.enabledPeer) cudaDeviceDisablePeerAccess(parts[1 - rank].device);
            // Host staging records transfer completion on the destination GPU.
            if (transport == Transport::Host && parts[1 - rank].device >= 0)
                cudaSetDevice(parts[1 - rank].device);
            for (auto event : part.transferred) if (event) cudaEventDestroy(event);
        }
        if (hostProfiles) cudaFreeHost(hostProfiles);
        if (parts[0].device >= 0) cudaSetDevice(parts[0].device);
    }

public:
    Solver(std::array<int, 2> devices, const std::string &requested, bool testMode = false) : emulated(testMode)
    {
        try
        {
            if (moment_io::distributed()) throw std::runtime_error("only one distributed solver may be active");
            int count = 0;
            check(cudaGetDeviceCount(&count), "enumerate GPUs");
            if (devices[0] < 0 || devices[1] < 0 || devices[0] >= count || devices[1] >= count)
                throw std::runtime_error("two visible GPUs are required; use --devices with valid CUDA ordinals");
            if (devices[0] == devices[1] && !emulated)
                throw std::runtime_error("two-GPU execution requires distinct devices");
#if !defined(MRL_MULTI_GPU_TESTING)
            if (emulated) throw std::runtime_error("two-slab emulation is available only in test builds");
#endif
            if (emulated && devices[0] != devices[1])
                throw std::runtime_error("test emulation requires the same device for both slabs");
            parts[0].device = devices[0];
            parts[1].device = devices[1];
            if (requested != "peer" && requested != "host" && requested != "auto")
                throw std::runtime_error("halo transport must be peer, host or auto");

            int access[2] = {};
            if (!emulated)
            {
                check(cudaDeviceCanAccessPeer(&access[0], devices[0], devices[1]), "query forward peer access");
                check(cudaDeviceCanAccessPeer(&access[1], devices[1], devices[0]), "query reverse peer access");
            }
            const bool peer = access[0] && access[1];
            if (!emulated && requested == "peer" && !peer)
                throw std::runtime_error("bidirectional CUDA P2P is unavailable for these PCIe GPUs; select --halo-transport host explicitly, or choose a peer-capable pair");
            const bool staged = requested == "host" || (!emulated && requested == "auto" && !peer);
            transport = staged ? Transport::Host : emulated ? Transport::Local : Transport::Peer;

            ranks([&](const auto Rank)
            {
                auto &part = parts[Rank];
                part.device = devices[Rank];
                part.begin = Slab<Rank>::begin;
                part.planes = Slab<Rank>::planes;
                part.cells = Slab<Rank>::cells;
                check(cudaSetDevice(part.device), "select slab GPU");
                cudaDeviceProp properties;
                check(cudaGetDeviceProperties(&properties, part.device), "query slab GPU");
                part.name = properties.name;
                if (transport == Transport::Peer)
                {
                    const auto error = cudaDeviceEnablePeerAccess(devices[1 - Rank], 0);
                    if (error == cudaErrorPeerAccessAlreadyEnabled)
                    {
                        cudaGetLastError();
                    }
                    else
                    {
                        check(error, "enable peer access");
                        part.enabledPeer = true;
                    }
                }

                const size_t bytes = size_t(NUM_MOMENTS) * part.cells * sizeof(real_t);
                const size_t profileBytes = size_t(rbc::NS) * part.planes * sizeof(double);
                size_t free = 0, total = 0;
                check(cudaMemGetInfo(&free, &total), "query slab memory");
                const size_t required = 2 * bytes + 4 * haloBytes + (rbc::TILES + 1) * profileBytes;
                if (free < required)
                {
                    throw std::runtime_error("insufficient free GPU memory for local moment buffers and halos");
                }
                int low = 0, high = 0;
                check(cudaDeviceGetStreamPriorityRange(&low, &high), "query stream priorities");
                check(cudaStreamCreateWithPriority(&part.bulk, cudaStreamNonBlocking, low), "create bulk stream");
                check(cudaStreamCreateWithPriority(&part.skin, cudaStreamNonBlocking, high), "create skin stream");
                check(cudaStreamCreateWithPriority(&part.copy, cudaStreamNonBlocking, high), "create copy stream");
                check(cudaStreamCreateWithPriority(&part.upload, cudaStreamNonBlocking, high), "create upload stream");
                for (int i = 0; i < 2; ++i)
                {
                    check(cudaMalloc(&part.state[i], bytes), "allocate local moments");
                    check(cudaMalloc(&part.send[i], haloBytes), "allocate packed halo");
                    check(cudaMalloc(&part.halo[i], haloBytes), "allocate incoming halo");
                    check(cudaEventCreateWithFlags(&part.bulkDone[i], cudaEventDisableTiming), "create bulk event");
                    check(cudaEventCreateWithFlags(&part.skinDone[i], cudaEventDisableTiming), "create skin event");
                    check(cudaEventCreateWithFlags(&part.downloaded[i], cudaEventDisableTiming), "create staging event");
                    if (transport == Transport::Host)
                    {
                        check(cudaHostAlloc(&part.pinned[i], haloBytes, cudaHostAllocPortable), "allocate pinned halo staging");
                    }
                }
                check(cudaMalloc(&part.partial, rbc::TILES * profileBytes), "allocate local reduction tiles");
                check(cudaMalloc(&part.profile, profileBytes), "allocate local profiles");
                std::cout << "slab " << Rank << " GPU " << part.device << " " << part.name
                          << " z=[" << part.begin << ',' << part.begin + part.planes << ") allocated=" << required << " bytes\n";
            });
            for (int rank = 0; rank < 2; ++rank)
            {
                check(cudaSetDevice(parts[transport == Transport::Host ? 1 - rank : rank].device), "select transfer event device");
                for (auto &event : parts[rank].transferred)
                    check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "create transfer event");
            }
            check(cudaHostAlloc(&hostProfiles, rbc::PROFILE_SIZE * sizeof(double), cudaHostAllocPortable), "allocate pinned reduced profiles");
            primary();
            moment_io::distributed() = this;
            std::cout << "two-slab Z decomposition, transport=" << transportName()
                      << " bytes/direction/step=" << haloBytes << '\n';
        }
        catch (...)
        {
            release();
            throw;
        }
    }

    ~Solver() override
    {
        release();
    }
    Solver(const Solver &) = delete;
    Solver &operator=(const Solver &) = delete;

    real_t *current() const {return parts[0].state[parity];}
    real_t *previous() const {return parts[0].state[1 - parity];}

    std::string transportName() const override
    {
        if (transport == Transport::Local) return "local-test";
        if (transport == Transport::Host) return emulated ? "host-staged-test" : "host-staged";
        return "cudaMemcpyPeerAsync";
    }

    std::string deviceNames() const override
    {
        return std::to_string(parts[0].device) + ":" + parts[0].name + " z=[0," + std::to_string(SPLIT_Z) + ");" +
            std::to_string(parts[1].device) + ":" + parts[1].name + " z=[" + std::to_string(SPLIT_Z) + "," + std::to_string(NZ) + ")";
    }

    int physicalDeviceCount() const override
    {
        return emulated ? 1 : 2;
    }

    void initialize()
    {
        ranks([&](const auto Rank)
        {
            auto &part = parts[Rank];
            check(cudaSetDevice(part.device), "select initialization GPU");
            two_gpu::initialize<Rank><<<dim3(GRID_X, GRID_Y, (part.planes + BLOCK_NZ - 1) / BLOCK_NZ), dim3(BLOCK_NX, BLOCK_NY, BLOCK_NZ), 0, part.bulk>>>(part.state[0]);
            initializePlate<Rank><<<dim3((NX + 31) / 32, (NY + 3) / 4), dim3(32, 4), 0, part.bulk>>>(part.state[0]);
            check(cudaGetLastError(), "initialize local slab");
        });
        prime();
    }

    // Also used after restoring a decomposition-independent checkpoint. Waiting
    // once here completes any pageable default-stream H2D restore copies.
    void prime()
    {
        synchronize();
        ranks([&](const auto Rank)
        {
            auto &part = parts[Rank];
            check(cudaSetDevice(part.device), "select initial halo GPU");
            packInitial<Rank><<<(STRIDE + 255) / 256, 256, 0, part.bulk>>>(part.state[parity], part.send[parity]);
            check(cudaEventRecord(part.skinDone[parity], part.bulk), "record initial skin");
            check(cudaEventRecord(part.bulkDone[parity], part.bulk), "record initial bulk");
            check(cudaGetLastError(), "pack initial halo");
        });
        exchange(parity, 1 - parity);
    }

    void advance(natural_t step)
    {
        const int old = parity, next = 1 - old;
        // Old-generation events were recorded while enqueueing the previous
        // step. Each wait captures that recording before its slot is reused.
        ranks([&](const auto Rank)
        {
            auto &part = parts[Rank];
            auto &peer = parts[1 - Rank];
            check(cudaSetDevice(part.device), "select timestep GPU");
            check(cudaStreamWaitEvent(part.skin, part.bulkDone[old], 0), "wait donor bulk");
            check(cudaStreamWaitEvent(part.skin, peer.transferred[old], 0), "wait incoming halo");
            check(cudaStreamWaitEvent(part.skin, part.transferred[next], 0), "protect packed halo reuse");
            skin<Rank><<<dim3((NX + 31) / 32, (NY + 3) / 4), dim3(32, 4), 0, part.skin>>>(
                Read<Rank, true>{part.state[old], part.halo[old]}, Write<Rank, true>{part.state[next], part.send[next]}, step);
            check(cudaEventRecord(part.skinDone[next], part.skin), "record collided skin");
            exchangeOne(Rank, next, old, true);

            check(cudaStreamWaitEvent(part.bulk, part.skinDone[old], 0), "protect local input/output reuse");
            const natural_t planes = part.planes - 1;
            interior<Rank><<<dim3(GRID_X, GRID_Y, (planes + BLOCK_NZ - 1) / BLOCK_NZ), dim3(BLOCK_NX, BLOCK_NY, BLOCK_NZ), 0, part.bulk>>>(
                Read<Rank>{part.state[old], nullptr}, Write<Rank>{part.state[next], nullptr}, step);
            check(cudaEventRecord(part.bulkDone[next], part.bulk), "record collided bulk");
            check(cudaGetLastError(), "advance local slab");
        });
        // The next step selects rank zero itself. Avoid an otherwise redundant
        // cudaSetDevice here; every sampled/output operation selects its GPU.
        parity = next;
    }

    void synchronize()
    {
        for (const auto &part : parts)
        {
            check(cudaSetDevice(part.device), "select synchronization GPU");
            check(cudaDeviceSynchronize(), "complete local GPU work");
        }
        primary();
    }

    void sample(rbc::Statistics &statistics, natural_t step)
    {
        ranks([&](const auto Rank)
        {
            auto &part = parts[Rank];
            check(cudaSetDevice(part.device), "select statistics GPU");
            check(cudaStreamWaitEvent(part.bulk, part.skinDone[parity], 0), "wait sampled skin");
            check(cudaStreamWaitEvent(part.bulk, parts[1 - Rank].transferred[parity], 0), "wait sampled gradient halo");
            two_gpu::reduceTiles<Rank><<<dim3(rbc::TILES, part.planes), rbc::THREADS, 0, part.bulk>>>(Read<Rank>{part.state[parity], part.halo[parity]}, part.partial, step);
            two_gpu::reducePlanes<Rank><<<(rbc::NS * part.planes + rbc::WARPS - 1) / rbc::WARPS, rbc::THREADS, 0, part.bulk>>>(part.partial, part.profile);
            check(cudaMemcpyAsync(hostProfiles + size_t(part.begin) * rbc::NS, part.profile, size_t(part.planes) * rbc::NS * sizeof(double), cudaMemcpyDeviceToHost, part.bulk), "copy reduced local profiles");
            check(cudaGetLastError(), "reduce local statistics");
        });
        for (const auto &part : parts)
        {
            check(cudaSetDevice(part.device), "select profile completion GPU");
            check(cudaStreamSynchronize(part.bulk), "finish sampled profiles");
        }
        std::copy(hostProfiles, hostProfiles + rbc::PROFILE_SIZE, statistics.profile.begin());
        primary();
        statistics.sampleProfile(step);
    }

    void download(real_t *host, const real_t *token, natural_t field, natural_t first, natural_t count) override
    {
        const int index = buffer(token);
        for (const auto &part : parts)
        {
            const natural_t begin = std::max(first, part.begin * STRIDE);
            const natural_t end = std::min(first + count, (part.begin + part.planes) * STRIDE);
            if (begin >= end) continue;
            check(cudaSetDevice(part.device), "select checkpoint source");
            check(cudaMemcpy(host + begin - first, part.state[index] + size_t(field) * part.cells + begin - part.begin * STRIDE,
                             size_t(end - begin) * sizeof(real_t), cudaMemcpyDeviceToHost), "download local checkpoint range");
        }
        primary();
    }

    void upload(real_t *token, const real_t *host, natural_t field, natural_t first, natural_t count) override
    {
        const int index = buffer(token);
        for (const auto &part : parts)
        {
            const natural_t begin = std::max(first, part.begin * STRIDE);
            const natural_t end = std::min(first + count, (part.begin + part.planes) * STRIDE);
            if (begin >= end) continue;
            check(cudaSetDevice(part.device), "select checkpoint destination");
            check(cudaMemcpy(part.state[index] + size_t(field) * part.cells + begin - part.begin * STRIDE, host + begin - first,
                             size_t(end - begin) * sizeof(real_t), cudaMemcpyHostToDevice), "restore local checkpoint range");
        }
        primary();
    }

    void extract(real_t *host, const real_t *token, natural_t index, bool xz, natural_t step) override
    {
        const int state = buffer(token);
        ranks([&](const auto Rank)
        {
            auto &part = parts[Rank];
            if (!xz && (index < part.begin || index >= part.begin + part.planes)) return;
            check(cudaSetDevice(part.device), "select field output GPU");
            const size_t capacity = 4 * size_t(NX) * std::max(NY, part.planes);
            if (!part.output)
            {
                check(cudaMalloc(&part.output, capacity * sizeof(real_t)), "allocate local output slab");
                part.outputHost.resize(capacity);
            }
            const size_t plane = size_t(NX) * (xz ? part.planes : NY);
            extractPlane<Rank><<<(plane + 255) / 256, 256, 0, part.bulk>>>(part.state[state], part.output, index, xz, step);
            check(cudaStreamSynchronize(part.bulk), "finish local field extraction");
            check(cudaMemcpy(part.outputHost.data(), part.output, 4 * plane * sizeof(real_t), cudaMemcpyDeviceToHost), "copy local field output");
            for (int field = 0; field < 4; ++field)
            {
                const size_t globalPlane = xz ? size_t(NX) * NZ : plane;
                const size_t offset = xz ? size_t(NX) * part.begin : 0;
                std::copy_n(part.outputHost.data() + field * plane, plane, host + field * globalPlane + offset);
            }
        });
        primary();
    }

    double captureTemperature(real_t *host, const real_t *token) override
    {
        const int state = buffer(token);
        for (auto &part : parts)
        {
            check(cudaSetDevice(part.device), "select temperature capture GPU");
            if (!part.snapshotCopy)
            {
                check(cudaStreamCreateWithFlags(&part.snapshotCopy, cudaStreamNonBlocking), "create temperature capture stream");
                check(cudaEventCreate(&part.snapshotBegin), "create temperature capture start");
                check(cudaEventCreate(&part.snapshotEnd), "create temperature capture end");
            }
            {
                h5out::ComputeWaitTimer computation;
                check(cudaEventSynchronize(part.bulkDone[state]), "wait captured interior");
                check(cudaEventSynchronize(part.skinDone[state]), "wait captured interface");
            }
            check(cudaEventRecord(part.snapshotBegin, part.snapshotCopy), "start temperature capture");
            check(cudaMemcpyAsync(host + size_t(part.begin) * STRIDE, part.state[state] + size_t(TEMP) * part.cells,
                                  size_t(part.cells) * sizeof(real_t), cudaMemcpyDeviceToHost, part.snapshotCopy), "capture local temperature");
            check(cudaEventRecord(part.snapshotEnd, part.snapshotCopy), "record local temperature capture");
        }
        double elapsed = 0;
        for (const auto &part : parts)
        {
            check(cudaSetDevice(part.device), "select temperature capture completion GPU");
            check(cudaEventSynchronize(part.snapshotEnd), "finish local temperature capture");
            float milliseconds = 0;
            check(cudaEventElapsedTime(&milliseconds, part.snapshotBegin, part.snapshotEnd), "time local temperature capture");
            elapsed += milliseconds / 1000.;
        }
        primary();
        return elapsed;
    }

    void validation(double *host, const real_t *now, const real_t *old) override
    {
#if defined(RBC_VALIDATION_CASE)
        const int current = buffer(now), previous = buffer(old);
        ranks([&](const auto Rank)
        {
            auto &part = parts[Rank];
            check(cudaSetDevice(part.device), "select validation output GPU");
            const size_t plane = size_t(NX) * part.planes;
            const size_t count = (2 * NUM_MOMENTS + 1) * plane;
            if (!part.validation)
            {
                check(cudaMalloc(&part.validation, count * sizeof(double)), "allocate local validation slice");
                part.validationHost.resize(count);
            }
            extractValidation<Rank><<<dim3((NX + 127) / 128, part.planes), 128, 0, part.bulk>>>(part.state[current], part.state[previous], part.validation);
            check(cudaStreamSynchronize(part.bulk), "finish local validation slice");
            check(cudaMemcpy(part.validationHost.data(), part.validation, count * sizeof(double), cudaMemcpyDeviceToHost), "copy local validation slice");
            for (natural_t field = 0; field < 2 * NUM_MOMENTS + 1; ++field)
                std::copy_n(part.validationHost.data() + field * plane, plane, host + size_t(field) * NX * NZ + size_t(part.begin) * NX);
        });
        primary();
#else
        throw std::runtime_error("validation output is not compiled into this executable");
#endif
    }
};
}
