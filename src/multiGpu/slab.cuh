#pragma once

#include "../momentStorage.cuh"

namespace two_gpu
{
#if defined(MRL_SPLIT_Z)
constexpr natural_t SPLIT_Z = MRL_SPLIT_Z;
#else
constexpr natural_t SPLIT_Z = NZ / 2;
#endif
static_assert(SPLIT_Z >= 2 && SPLIT_Z <= NZ - 2, "each Z slab must own at least two planes");

template <int Rank>
struct Slab
{
    static_assert(Rank == 0 || Rank == 1);
    static constexpr natural_t begin = Rank == 0 ? 0 : SPLIT_Z;
    static constexpr natural_t end = Rank == 0 ? SPLIT_Z : NZ;
    static constexpr natural_t planes = end - begin;
    static constexpr natural_t cells = planes * STRIDE;
    static constexpr natural_t interface = Rank == 0 ? end - 1 : begin;
};

template <int Rank>
struct LocalRead
{
    const real_t *data;
};

struct HaloRead
{
    const real_t *data;
};

// Bounded reads are needed only by sampled gradient diagnostics. Bulk and skin
// population pulls use separate compile-time views without per-load branches.
template <int Rank, bool Interface = false>
struct Read
{
    const real_t *data;
    const real_t *halo;
};

template <int Rank, bool Pack = false>
struct Write
{
    real_t *data;
    real_t *send;
};

template <int Rank>
__device__ __forceinline__ real_t loadMoment(LocalRead<Rank> moments, natural_t index, natural_t field)
{
    return __ldg(moments.data + size_t(field) * Slab<Rank>::cells + (index - Slab<Rank>::begin * STRIDE));
}

__device__ __forceinline__ real_t loadMoment(HaloRead moments, natural_t index, natural_t field)
{
    return __ldg(moments.data + size_t(field) * STRIDE + index % STRIDE);
}

template <int Rank, bool Interface>
__device__ __forceinline__ real_t loadMoment(Read<Rank, Interface> moments, natural_t index, natural_t field)
{
    if constexpr (Rank == 0)
    {
        if (index >= Slab<Rank>::end * STRIDE)
            return loadMoment(HaloRead{moments.halo}, index, field);
    }
    else
    {
        if (index < Slab<Rank>::begin * STRIDE)
            return loadMoment(HaloRead{moments.halo}, index, field);
    }
    return loadMoment(LocalRead<Rank>{moments.data}, index, field);
}

template <natural_t Q, int Rank, bool Interface>
__device__ __forceinline__ auto directionMoments(Read<Rank, Interface> moments)
{
    constexpr int cz = VelocitySet::cz<Q>();
    if constexpr (Interface && ((Rank == 0 && cz == -1) || (Rank == 1 && cz == 1)))
    {
        return HaloRead{moments.halo};
    }
    else
    {
        return LocalRead<Rank>{moments.data};
    }
}

template <int Rank, bool Pack>
__device__ __forceinline__ void storeMoment(Write<Rank, Pack> moments, natural_t index, natural_t field, real_t value)
{
    moments.data[size_t(field) * Slab<Rank>::cells + index - Slab<Rank>::begin * STRIDE] = value;
    if constexpr (Pack)
    {
        moments.send[size_t(field) * STRIDE + index % STRIDE] = value;
    }
}

template <int Rank>
__device__ __forceinline__ real_t &momentRef(Write<Rank, false> moments, natural_t index, natural_t field)
{
    return moments.data[size_t(field) * Slab<Rank>::cells + index - Slab<Rank>::begin * STRIDE];
}

template <int Rank, bool Pack>
__device__ __forceinline__ auto plateStorage(Write<Rank, Pack> moments)
{
    // A skin donor can also own a physical plate on very small validation grids.
    // Plate stores must never replace the packed interface plane.
    return Write<Rank, false>{moments.data, nullptr};
}
}
