#pragma once

#include "hdf5Output.cuh"
#include <zlib.h>
#include <array>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <csignal>
#include <ctime>
#include <thread>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <unistd.h>

// CPU-only chunk encoder and isolated writer. The HDF5 file retains the usual
// shuffle -> deflate -> Fletcher32 pipeline; no plugin or reader change is used.
namespace snapshot
{
using Clock = std::chrono::steady_clock;
inline double seconds(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}
inline double threadCpuSeconds()
{
    timespec value{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value)) throw std::runtime_error("query encoder CPU clock");
    return double(value.tv_sec) + value.tv_nsec * 1.e-9;
}

struct Request
{
    uint64_t step = 0, cells = 0;
    uint32_t nx = 0, ny = 0, nz = 0, precision = 0;
    uint32_t compression = 1, workers = 4, slot = 0, chunkY = 64, chunkX = 128;
    char path[4096] = {}, runId[256] = {};
};

struct Reply
{
    uint64_t step = 0, fileBytes = 0;
    double encodeWall = 0, encodeCpu = 0, write = 0, flush = 0, total = 0;
    int status = 0;
    char error[512] = {};
};

struct Encoded
{
    std::vector<unsigned char> raw, shuffled, bytes;
    hsize_t offset[3] = {};
    double cpu = 0;
};

struct Deflater
{
    z_stream stream = {};
    int level = -1;
    ~Deflater() {if (level >= 0) deflateEnd(&stream);}
    void compress(const std::vector<unsigned char> &input, std::vector<unsigned char> &output, int requested)
    {
        if (level != requested)
        {
            if (level >= 0) deflateEnd(&stream);
            stream = z_stream{};level = -1;
            if (deflateInit(&stream, requested) != Z_OK) throw std::runtime_error("initialize snapshot compressor");
            level = requested;
        }
        else if (deflateReset(&stream) != Z_OK) throw std::runtime_error("reset snapshot compressor");
        output.resize(compressBound(input.size()));
        stream.next_in = const_cast<Bytef *>(input.data());stream.avail_in = input.size();
        stream.next_out = output.data();stream.avail_out = output.size();
        if (deflate(&stream, Z_FINISH) != Z_STREAM_END) throw std::runtime_error("snapshot zlib compression failed");
        output.resize(stream.total_out);
    }
};

inline void checksum(std::vector<unsigned char> &data)
{
    // HDF5 Fletcher32: big-endian 16-bit words, little-endian trailer. Fold
    // bounded groups with 64-bit intermediates, including a final odd byte.
    uint64_t a = 0, b = 0;
    auto fold = [](uint64_t value)
    {
        while (value >> 16) value = (value & 65535) + (value >> 16);
        return value;
    };
    for (size_t first = 0; first < data.size(); first += 1024)
    {
        const size_t end = std::min(first + 1024, data.size());
        for (size_t i = first; i < end; i += 2)
        {
            a += uint64_t(data[i]) * 256 + (i + 1 < data.size() ? data[i + 1] : 0);
            b += a;
        }
        a = fold(a);
        b = fold(b);
    }
    const uint32_t value = uint32_t((b << 16) | a);
    for (int i = 0; i < 4; ++i) data.push_back(static_cast<unsigned char>(value >> (8 * i)));
}

inline void encode(Encoded &out, const unsigned char *input, const Request &job, size_t chunk, Deflater &compressor)
{
    static_assert(std::endian::native == std::endian::little, "raw HDF5 output requires a little-endian host");
    const double start = threadCpuSeconds();
    const size_t rows = (job.ny + job.chunkY - 1) / job.chunkY;
    const size_t cols = (job.nx + job.chunkX - 1) / job.chunkX;
    const size_t z = chunk / (rows * cols), y = chunk / cols % rows * job.chunkY, x = chunk % cols * job.chunkX;
    out.offset[0] = z;out.offset[1] = y;out.offset[2] = x;
    const size_t cells = size_t(job.chunkY) * job.chunkX, bytes = cells * job.precision;
    out.raw.resize(bytes);
    if (y + job.chunkY > job.ny || x + job.chunkX > job.nx)
        std::fill(out.raw.begin(), out.raw.end(), 0);
    for (size_t j = 0; j < std::min<size_t>(job.chunkY, job.ny - y); ++j)
    {
        std::memcpy(out.raw.data() + j * job.chunkX * job.precision,
                    input + ((z * job.ny + y + j) * job.nx + x) * job.precision,
                    std::min<size_t>(job.chunkX, job.nx - x) * job.precision);
    }
    if (job.compression)
    {
        out.shuffled.resize(bytes);
        for (size_t d = 0; d < job.precision; ++d)
            for (size_t i = 0; i < cells; ++i)
                out.shuffled[d * cells + i] = out.raw[i * job.precision + d];
        compressor.compress(out.shuffled, out.bytes, job.compression);
    }
    else
    {
        out.bytes.assign(out.raw.begin(), out.raw.end());
    }
    checksum(out.bytes);
    out.cpu = threadCpuSeconds() - start;
}

// Persistent bounded workers. Only the caller touches HDF5; encoder threads
// write independent buffers, so neither HDF5 thread safety nor parallel HDF5
// is required. Buffers survive batches and subsequent snapshots.
class Encoder
{
    std::mutex mutex;
    std::condition_variable ready, done;
    std::vector<std::thread> threads;
    std::atomic<size_t> next{0};
    size_t generation = 0, remaining = 0, first = 0, count = 0;
    bool stopping = false;
    const unsigned char *input = nullptr;
    Request job;
    std::exception_ptr failure;

    void work()
    {
        size_t seen = 0;
        Deflater compressor;
        for (;;)
        {
            std::unique_lock lock(mutex);
            ready.wait(lock, [&] {return stopping || seen != generation;});
            if (stopping) return;
            seen = generation;
            lock.unlock();
            try
            {
                for (size_t i = next.fetch_add(1); i < count; i = next.fetch_add(1))
                    encode(chunks[i], input, job, first + i, compressor);
            }
            catch (...)
            {
                std::lock_guard guard(mutex);
                if (!failure) failure = std::current_exception();
            }
            lock.lock();
            if (--remaining == 0) done.notify_one();
        }
    }

public:
    std::vector<Encoded> chunks;
    explicit Encoder(unsigned workers) : chunks(std::max(1u, workers) * 16)
    {
        if (!workers || workers > 32) throw std::runtime_error("snapshot workers must be 1..32");
        try
        {
            for (unsigned i = 0; i < workers; ++i) threads.emplace_back([this] {work();});
        }
        catch (...)
        {
            stop();
            throw;
        }
    }
    void stop()
    {
        {std::lock_guard lock(mutex);stopping = true;}
        ready.notify_all();
        for (auto &thread : threads) if (thread.joinable()) thread.join();
    }
    ~Encoder() {stop();}

    size_t batch(const unsigned char *values, const Request &request, size_t begin, size_t total)
    {
        std::unique_lock lock(mutex);
        input = values;job = request;first = begin;
        count = std::min(chunks.size(), total - begin);
        failure = nullptr;next = 0;remaining = threads.size();++generation;
        ready.notify_all();
        done.wait(lock, [&] {return remaining == 0;});
        if (failure) std::rethrow_exception(failure);
        return count;
    }
};

inline Reply write(const unsigned char *input, const Request &job, Encoder &encoder)
{
    const auto start = Clock::now();
    Reply result;result.step = job.step;
    const std::filesystem::path path(job.path);
    auto temporary = path;temporary += ".tmp";
    if (std::filesystem::exists(path)) throw std::runtime_error("refusing to overwrite snapshot " + path.string());
    h5out::compression() = job.compression;
    {
        h5out::Handle file(H5Fcreate(temporary.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
        h5out::attribute(file, "step", std::to_string(job.step));
        h5out::attribute(file, "run_id", job.runId);
        h5out::attribute(file, "velocity_convention", "physical_u");
        h5out::attribute(file, "axis_order", "zyx");
        h5out::attribute(file, "lattice_spacing", "1");
        h5out::attribute(file, "wall_nodes", "0,NZ-1");
        h5out::Handle group(h5out::group(file, "fields"), H5Gclose);
        const hid_t disk = job.precision == 4 ? H5T_IEEE_F32LE : H5T_IEEE_F64LE;
        h5out::Handle data(h5out::dataset(group, "temperature", disk, {job.nz, job.ny, job.nx},
                                        {1, job.chunkY, job.chunkX}), H5Dclose);
        const size_t count = size_t(job.nz) * ((job.ny + job.chunkY - 1) / job.chunkY) * ((job.nx + job.chunkX - 1) / job.chunkX);
        for (size_t first = 0; first < count;)
        {
            const auto encoding = Clock::now();
            const size_t batch = encoder.batch(input, job, first, count);
            result.encodeWall += seconds(encoding);
            const auto writing = Clock::now();
            for (size_t i = 0; i < batch; ++i)
            {
                const auto &chunk = encoder.chunks[i];
                result.encodeCpu += chunk.cpu;
                h5out::check(H5Dwrite_chunk(data, H5P_DEFAULT, 0, chunk.offset, chunk.bytes.size(), chunk.bytes.data()));
            }
            result.write += seconds(writing);
            first += batch;
        }
        const auto flushing = Clock::now();
        h5out::check(H5Dclose(data.id));data.id = -1;
        h5out::check(H5Gclose(group.id));group.id = -1;
        h5out::flush(file);
        h5out::check(H5Fclose(file.id));file.id = -1;
        result.flush = seconds(flushing);
    }
    std::filesystem::rename(temporary, path);
    result.total = seconds(start);
    result.fileBytes = std::filesystem::file_size(path);
    return result;
}

// Entered by exec of the same specialized solver, before any CUDA API call.
// The process has a separate HDF5 library/lock from the simulation's statistics.
inline int worker(int memory, int channel, size_t bytes, pid_t parent)
{
    prctl(PR_SET_PDEATHSIG, SIGTERM);
    if (getppid() != parent) return 1;
    void *mapping = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, memory, 0);
    if (mapping == MAP_FAILED) return 1;
    std::unique_ptr<Encoder> encoder;
    unsigned workers = 0;
    for (;;)
    {
        Request request;
        const auto received = recv(channel, &request, sizeof(request), MSG_WAITALL);
        if (received == 0) break;
        if (received != sizeof(request)) {munmap(mapping, bytes);return 1;}
        Reply reply;reply.step = request.step;
        try
        {
            if (!request.cells || (request.precision != 4 && request.precision != 8) ||
                request.cells != uint64_t(request.nx) * request.ny * request.nz ||
                request.cells > bytes / request.precision || request.slot != 0 ||
                !request.workers || request.workers > 32 || request.compression > 9 ||
                !request.chunkY || !request.chunkX || request.chunkY > request.ny || request.chunkX > request.nx ||
                request.path[sizeof(request.path) - 1] || request.runId[sizeof(request.runId) - 1])
                throw std::runtime_error("invalid snapshot writer request");
            if (workers != request.workers)
            {
                encoder = std::make_unique<Encoder>(request.workers);
                workers = request.workers;
            }
            reply = write(static_cast<const unsigned char *>(mapping), request, *encoder);
        }
        catch (const std::exception &error)
        {
            reply.status = 1;
            std::strncpy(reply.error, error.what(), sizeof(reply.error) - 1);
        }
        if (send(channel, &reply, sizeof(reply), MSG_NOSIGNAL) != sizeof(reply) || reply.status) break;
    }
    munmap(mapping, bytes);
    return 0;
}
}
