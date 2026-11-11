#pragma once

#include "snapshotWriter.hpp"
#include <spawn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>

extern char **environ;

namespace snapshot
{
inline std::string &mode() {static std::string value = "async";return value;}
inline unsigned &workers() {static unsigned value = 4;return value;}
// Benchmark-only overrides; production retains the established chunk geometry.
inline unsigned &chunkRows() {static unsigned value = 64;return value;}
inline unsigned &chunkColumns() {static unsigned value = 128;return value;}

class TemperatureOutput
{
    real_t *host = nullptr;
    size_t bytes = size_t(CELLS) * sizeof(real_t);
    int memory = -1, channel = -1, device = -1;
    pid_t child = -1;
    cudaStream_t copy = nullptr;
    cudaEvent_t ready = nullptr, beginCopy = nullptr, endCopy = nullptr;
    bool registered = false, busy = false;
    std::unique_ptr<Encoder> encoder;
    Request pending;
    Clock::time_point submitted;
    double captureWall = 0, copyGpu = 0, backpressure = 0;
    double writerSeconds = 0;

    static void systemCheck(bool ok, const char *operation)
    {
        if (!ok) throw std::runtime_error(std::string(operation) + ": " + std::strerror(errno));
    }

    void launch()
    {
        int sockets[2] = {-1, -1};
        systemCheck(socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, sockets) == 0, "snapshot socketpair");
        // Duplicate sources above the child's fixed descriptors before dup2,
        // so no spawn action can accidentally overwrite a later source fd.
        const int shared = fcntl(memory, F_DUPFD_CLOEXEC, 128);
        const int connection = fcntl(sockets[1], F_DUPFD_CLOEXEC, 128);
        if (shared < 0 || connection < 0)
        {
            if (shared >= 0) close(shared);
            if (connection >= 0) close(connection);
            close(sockets[0]);close(sockets[1]);
            throw std::runtime_error("cannot duplicate snapshot worker descriptors");
        }
        posix_spawn_file_actions_t actions;
        int error = posix_spawn_file_actions_init(&actions);
        const bool initialized = error == 0;
        if (!error) error = posix_spawn_file_actions_adddup2(&actions, shared, 3);
        if (!error) error = posix_spawn_file_actions_adddup2(&actions, connection, 4);
        const auto size = std::to_string(bytes), parent = std::to_string(getpid());
        std::array<char *, 5> args{const_cast<char *>("thermal-mrlbm"), const_cast<char *>("--snapshot-worker"),
                                   const_cast<char *>(size.c_str()), const_cast<char *>(parent.c_str()), nullptr};
        if (!error) error = posix_spawn(&child, "/proc/self/exe", &actions, nullptr, args.data(), environ);
        if (initialized) posix_spawn_file_actions_destroy(&actions);
        close(shared);close(connection);close(sockets[1]);
        if (error)
        {
            close(sockets[0]);child = -1;
            throw std::runtime_error(std::string("cannot spawn snapshot writer: ") + std::strerror(error));
        }
        channel = sockets[0];
    }

    void allocate()
    {
        if (host) return;
        moment_io::check(cudaGetDevice(&device), "query snapshot GPU");
        moment_io::check(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking), "create snapshot copy stream");
        moment_io::check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), "create snapshot readiness event");
        moment_io::check(cudaEventCreate(&beginCopy), "create snapshot timing event");
        moment_io::check(cudaEventCreate(&endCopy), "create snapshot timing event");
        memory = int(syscall(SYS_memfd_create, "thermal-temperature", MFD_CLOEXEC));
        systemCheck(memory >= 0, "create snapshot shared memory");
        systemCheck(ftruncate(memory, bytes) == 0, "size snapshot shared memory");
        void *mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, memory, 0);
        systemCheck(mapping != MAP_FAILED, "map snapshot shared memory");
        host = static_cast<real_t *>(mapping);
        const auto error = cudaHostRegister(host, bytes, cudaHostRegisterPortable);
        if (error != cudaSuccess)
            throw std::runtime_error("cannot pin temperature snapshot; use --field-io serial for bounded plane staging: " + std::string(cudaGetErrorString(error)));
        registered = true;
        if (mode() == "async") launch();
        else encoder = std::make_unique<Encoder>(workers());
    }

    void record(const Reply &reply)
    {
        busy = false;
        if (reply.step != pending.step) throw std::runtime_error("snapshot completion order mismatch");
        if (reply.status) throw std::runtime_error(std::string("snapshot writer failed: ") + reply.error);
        writerSeconds += reply.total;
        // This table is additional timing data. Existing scientific tables,
        // runtime columns, checkpoint accumulators and readers are unchanged.
        h5out::OutputTimer timer;
        h5out::Handle file(h5out::file(statisticsPath()), H5Fclose);
        h5out::Handle group(h5out::group(file, "statistics"), H5Gclose);
        h5out::append(group, "exports",
            "step,payload_bytes,packing_s,d2h_gpu_s,capture_wall_s,backpressure_s,compression_wall_s,compression_cpu_s,chunk_write_s,flush_s,writer_s,file_bytes,completion_latency_s,workers,compression",
            {double(reply.step), double(bytes), 0, copyGpu, captureWall, backpressure,
             reply.encodeWall, reply.encodeCpu, reply.write, reply.flush, reply.total, double(reply.fileBytes), seconds(submitted),
             double(pending.workers), double(pending.compression)});
        h5out::flush(file);
    }

public:
    ~TemperatureOutput()
    {
        // Normal completion drains explicitly before reporting elapsed time.
        // On an exception, stop the child before releasing its shared buffer.
        if (child > 0)
        {
            if (busy) kill(child, SIGTERM);
            if (channel >= 0) {close(channel);channel = -1;}
            int status = 0;
            while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
        }
        if (channel >= 0) close(channel);
        if (device >= 0) cudaSetDevice(device);
        if (copy) cudaStreamSynchronize(copy);
        if (registered) cudaHostUnregister(host);
        if (host) munmap(host, bytes);
        if (memory >= 0) close(memory);
        if (copy) cudaStreamDestroy(copy);
        if (ready) cudaEventDestroy(ready);
        if (beginCopy) cudaEventDestroy(beginCopy);
        if (endCopy) cudaEventDestroy(endCopy);
    }

    void poll(bool wait = false)
    {
        if (!busy) return;
        Reply reply;
        ssize_t count;
        do {count = recv(channel, &reply, sizeof(reply), wait ? 0 : MSG_DONTWAIT);} while (count < 0 && errno == EINTR);
        if (count < 0 && !wait && (errno == EAGAIN || errno == EWOULDBLOCK)) return;
        if (count != sizeof(reply)) throw std::runtime_error("snapshot writer exited without a complete acknowledgement");
        record(reply);
    }

    void drain() {poll(true);}
    double backgroundSeconds() const {return writerSeconds;}

    // Public direct-temperature entry also permits representative 512^3 IO
    // benchmarks without allocating the solver's fourteen moment fields.
    void save(const real_t *moments, natural_t step, bool directTemperature = false)
    {
        const auto start = Clock::now();
        allocate();
        poll();
        const auto waiting = Clock::now();
        drain();
        backpressure = seconds(waiting);
        pending = Request{};
        pending.step = step;pending.cells = CELLS;
        pending.nx = NX;pending.ny = NY;pending.nz = NZ;pending.precision = sizeof(real_t);
        pending.compression = h5out::compression();pending.workers = workers();
        pending.chunkY = std::min(chunkRows(), NY);pending.chunkX = std::min(chunkColumns(), NX);
        const auto dir = getSimulationHdf5Directory() / "fields";
        std::filesystem::create_directories(dir);
        const auto path = (dir / (outputStepName(step) + ".h5")).string();
        if (path.size() >= sizeof(pending.path) || simulationRunId().size() >= sizeof(pending.runId))
            throw std::runtime_error("snapshot path or run ID exceeds writer protocol limits");
        std::strcpy(pending.path, path.c_str());std::strcpy(pending.runId, simulationRunId().c_str());
        submitted = start;
        const auto capturing = Clock::now();
        if (moment_io::distributed() && !directTemperature)
        {
            copyGpu = moment_io::distributed()->captureTemperature(host, moments);
        }
        else
        {
            moment_io::check(cudaSetDevice(device), "select snapshot copy GPU");
            moment_io::check(cudaEventRecord(ready, nullptr), "record snapshot state readiness");
            {
                h5out::ComputeWaitTimer computation;
                moment_io::check(cudaEventSynchronize(ready), "wait snapshot state readiness");
            }
            moment_io::check(cudaEventRecord(beginCopy, copy), "start snapshot transfer timer");
            moment_io::check(cudaMemcpyAsync(host, directTemperature ? moments : moments + size_t(TEMP) * CELLS,
                                           bytes, cudaMemcpyDeviceToHost, copy), "capture temperature snapshot");
            moment_io::check(cudaEventRecord(endCopy, copy), "finish snapshot transfer timer");
            // Release the solver's alternating state only after its temperature
            // has been captured. CPU compression/writes continue independently.
            moment_io::check(cudaEventSynchronize(endCopy), "finish temperature capture");
            float milliseconds = 0;
            moment_io::check(cudaEventElapsedTime(&milliseconds, beginCopy, endCopy), "time temperature capture");
            copyGpu = milliseconds / 1000.;
        }
        captureWall = seconds(capturing);
        if (mode() == "async")
        {
            if (send(channel, &pending, sizeof(pending), MSG_NOSIGNAL) != sizeof(pending))
                throw std::runtime_error("cannot submit temperature snapshot to writer");
            busy = true;
        }
        else
        {
            record(write(reinterpret_cast<const unsigned char *>(host), pending, *encoder));
        }
    }
};
}
