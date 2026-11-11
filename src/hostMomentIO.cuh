#pragma once

#include "deviceFunctions.cuh"
#include <stdexcept>
#include <string>

namespace moment_io
{
inline void check(cudaError_t error, const char *operation)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

// Only scientific output/checkpoint calls use this bridge. Device kernels use
// compile-time storage views. One solver owns the bridge for its lifetime.
struct Distributed
{
    virtual ~Distributed() = default;
    virtual void download(real_t *, const real_t *, natural_t, natural_t, natural_t) = 0;
    virtual void upload(real_t *, const real_t *, natural_t, natural_t, natural_t) = 0;
    virtual void extract(real_t *, const real_t *, natural_t, bool, natural_t) = 0;
    virtual double captureTemperature(real_t *, const real_t *) = 0;
    virtual void validation(double *, const real_t *, const real_t *) = 0;
    virtual std::string transportName() const = 0;
    virtual std::string deviceNames() const = 0;
    virtual int physicalDeviceCount() const = 0;
};

inline Distributed *&distributed()
{
    static Distributed *instance = nullptr;
    return instance;
}

inline void download(real_t *host, const real_t *moments, natural_t field, natural_t first, natural_t count)
{
    if (distributed())
    {
        distributed()->download(host, moments, field, first, count);
    }
    else
    {
        check(cudaMemcpy(host, moments + size_t(field) * CELLS + first, size_t(count) * sizeof(real_t), cudaMemcpyDeviceToHost), "download moments");
    }
}

inline void upload(real_t *moments, const real_t *host, natural_t field, natural_t first, natural_t count)
{
    if (distributed())
    {
        distributed()->upload(moments, host, field, first, count);
    }
    else
    {
        check(cudaMemcpy(moments + size_t(field) * CELLS + first, host, size_t(count) * sizeof(real_t), cudaMemcpyHostToDevice), "upload moments");
    }
}
}
