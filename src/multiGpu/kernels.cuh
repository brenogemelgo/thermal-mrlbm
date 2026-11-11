#pragma once

#include "slab.cuh"
#include "../initialConditions.cuh"
#include "../kernel.cuh"
#include "../rbcStatistics.cuh"

namespace two_gpu
{
template <int Rank>
__global__ void initialize(real_t *moments)
{
    const natural_t x = blockIdx.x * BLOCK_NX + threadIdx.x;
    const natural_t y = blockIdx.y * BLOCK_NY + threadIdx.y;
    const natural_t z = Slab<Rank>::begin + blockIdx.z * BLOCK_NZ + threadIdx.z;
    if (z < Slab<Rank>::end)
    {
        initializeNode(Write<Rank>{moments, nullptr}, x, y, z);
    }
}

template <int Rank>
__global__ void initializePlate(real_t *moments)
{
    const natural_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const natural_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY)
    {
        return;
    }

    constexpr natural_t z = Rank == 0 ? 0 : NZ - 1;
    constexpr natural_t donorZ = Rank == 0 ? 1 : NZ - 2;
    const natural_t donor = global3(x, y, donorZ);
    const LocalRead<Rank> input{moments};
    const HydroState state = loadHydro(input, donor);
    const real_t temperature = loadMoment(input, donor, TEMP);
    const real_t flux[3] = {
        loadMoment(input, donor, TUX),
        loadMoment(input, donor, TUY),
        loadMoment(input, donor, TUZ)};
    real_t acceleration[3];
    nodeAcceleration(temperature, x, y, donorZ, 0, acceleration);
    writeNEEPlate(Write<Rank>{moments, nullptr}, global3(x, y, z), z, state, temperature, flux, acceleration);
}

template <int Rank>
__global__ void packInitial(const real_t *moments, real_t *send)
{
    const natural_t xy = blockIdx.x * blockDim.x + threadIdx.x;
    if (xy < STRIDE)
    {
        const natural_t index = Slab<Rank>::interface * STRIDE + xy;
        constexpr_for<0, NUM_MOMENTS>(
            [&](const auto Field) noexcept
            {
                send[size_t(Field) * STRIDE + xy] = loadMoment(LocalRead<Rank>{moments}, index, Field);
            });
    }
}

template <int Rank>
__global__ void skin(Read<Rank, true> moments, Write<Rank, true> buffer, natural_t step)
{
    const natural_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const natural_t y = blockIdx.y * blockDim.y + threadIdx.y;
    streamCollideNode(moments, buffer, x, y, Slab<Rank>::interface, step);
}

template <int Rank>
__global__ void interior(Read<Rank> moments, Write<Rank> buffer, natural_t step)
{
    constexpr natural_t begin = Rank == 0 ? Slab<Rank>::begin : Slab<Rank>::begin + 1;
    constexpr natural_t end = Rank == 0 ? Slab<Rank>::end - 1 : Slab<Rank>::end;
    const natural_t x = blockIdx.x * BLOCK_NX + threadIdx.x;
    const natural_t y = blockIdx.y * BLOCK_NY + threadIdx.y;
    const natural_t z = begin + blockIdx.z * BLOCK_NZ + threadIdx.z;
    if (z < end)
    {
        streamCollideNode(moments, buffer, x, y, z, step);
    }
}

template <int Rank>
__global__ void reduceTiles(Read<Rank> moments, double *partial, natural_t step)
{
    rbc::reduceTilesBody(moments, partial, step, Slab<Rank>::begin);
}

template <int Rank>
__global__ void reducePlanes(const double *partial, double *profiles)
{
    rbc::reducePlanesBody(partial, profiles, rbc::NS * Slab<Rank>::planes);
}

template <int Rank>
__global__ void extractPlane(const real_t *moments, real_t *packed, natural_t index, bool xz, natural_t step)
{
    const natural_t q = blockIdx.x * blockDim.x + threadIdx.x;
    const natural_t plane = NX * (xz ? Slab<Rank>::planes : NY);
    if (q >= plane)
    {
        return;
    }

    const natural_t x = q % NX;
    const natural_t y = xz ? index : q / NX;
    const natural_t z = xz ? Slab<Rank>::begin + q / NX : index;
    const natural_t node = global3(x, y, z);
    const LocalRead<Rank> input{moments};
    const real_t temperature = loadMoment(input, node, TEMP);
    real_t acceleration[3];
    nodeAcceleration(temperature, x, y, z, step, acceleration);
    packed[q] = temperature;
    constexpr_for<0, 3>(
        [&](const auto D) noexcept
        {
            packed[size_t(D + 1) * plane + q] = loadMoment(input, node, UX + D) - acceleration[D] / 2;
        });
}

#if defined(RBC_VALIDATION_CASE)
template <int Rank>
__global__ void extractValidation(const real_t *now, const real_t *old, double *slice)
{
    const natural_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const natural_t z = Slab<Rank>::begin + blockIdx.y;
    if (x >= NX)
    {
        return;
    }

    constexpr size_t plane = size_t(NX) * Slab<Rank>::planes;
    const size_t i = x + size_t(blockIdx.y) * NX;
    double mismatch = 0;
    constexpr_for<0, NUM_MOMENTS>(
        [&](const auto Field) noexcept
        {
            const double a = loadMoment(LocalRead<Rank>{now}, global3(x, 0, z), Field);
            const double b = loadMoment(LocalRead<Rank>{old}, global3(x, 0, z), Field);
            slice[size_t(Field) * plane + i] = a;
            slice[size_t(Field + NUM_MOMENTS) * plane + i] = b;
            for (natural_t y = 1; y < NY; ++y)
            {
                const double c = loadMoment(LocalRead<Rank>{now}, global3(x, y, z), Field);
                const double d = loadMoment(LocalRead<Rank>{old}, global3(x, y, z), Field);
                mismatch = !isfinite(c) || !isfinite(d) ? INFINITY : fmax(mismatch, fmax(fabs(c - a), fabs(d - b)));
            }
        });
    slice[2 * NUM_MOMENTS * plane + i] = mismatch;
}
#endif
}
