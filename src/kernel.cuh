#pragma once

#include "reconstruction.cuh"
#include "boundary/deviceRuntime.cuh"

template <natural_t Q, typename Input>
__device__ __forceinline__ void accumulateDirection(
    Input moments,
    natural_t x,
    natural_t y,
    natural_t z,
    HydroState &state,
    real_t &temperature,
    real_t (&flux)[3],
    natural_t step)
{
    constexpr int cx = VelocitySet::cx<Q>();
    constexpr int cy = VelocitySet::cy<Q>();
    constexpr int cz = VelocitySet::cz<Q>();

    const natural_t source = jetPullNeighborIndex(x, y, z, cx, cy, cz);
    const natural_t sourceX = source % NX;
    const natural_t sourceY = (source / NX) % NY;
    const natural_t sourceZ = source / STRIDE;
    const auto sourceMoments = directionMoments<Q>(moments);
    const HydroState donor = loadHydro(sourceMoments, source);
    const real_t donorTemperature = loadMoment(sourceMoments, source, TEMP);

    real_t acceleration[3];
    nodeAcceleration(donorTemperature, sourceX, sourceY, sourceZ, step, acceleration);

    const real_t f = hydroPopulation<Q>(donor, acceleration);
    state.rho += f;
    state.v[0] += cx * f;
    state.v[1] += cy * f;
    state.v[2] += cz * f;
    state.p[0] += VelocitySet::hxx<Q>() * f;
    state.p[1] += VelocitySet::hyy<Q>() * f;
    state.p[2] += VelocitySet::hzz<Q>() * f;
    state.p[3] += cx * cy * f;
    state.p[4] += cx * cz * f;
    state.p[5] += cy * cz * f;

    const real_t donorFlux[3] = {
        loadMoment(sourceMoments, source, TUX),
        loadMoment(sourceMoments, source, TUY),
        loadMoment(sourceMoments, source, TUZ)};

    // V2 stores physical T; the post-source zeroth moment is T + Q/2.
    const real_t g = thermalPopulation<Q>(
        donorTemperature + validation::scalarSource(sourceX, sourceZ, step) / 2,
        donorFlux);

    temperature += g;
    flux[0] += cx * g;
    flux[1] += cy * g;
    flux[2] += cz * g;
}

template <typename Input, typename Output>
__device__ __forceinline__ void streamCollideNode(
    Input moments,
    Output dbuffer,
    natural_t x,
    natural_t y,
    natural_t z,
    natural_t step)
{
    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

#if defined(CASE_RAYLEIGH_BENARD)
    // Each adjacent interior thread also owns its plate node. It uses the
    // freshly collided donor below, without reading another thread's output.
    if (z == 0 || z == NZ - 1)
    {
        return;
    }
#endif

    HydroState state{};
    real_t temperature = 0;
    real_t flux[3] = {};

#if defined(CASE_JET)
    const uint8_t type = boundaryMask(x, y, z);
    if (type != BULK)
    {
        dispatchIRBCBoundary(
            moments, x, y, z, step, type,
            state.rho, state.v[0], state.v[1], state.v[2],
            state.p[0], state.p[1], state.p[2], state.p[3], state.p[4], state.p[5],
            temperature, flux[0], flux[1], flux[2]);
        // The jet dispatcher returns A2/rho, without the isotropic cs2.
    }
    else
#endif
    {
        constexpr_for<0, VelocitySet::Q()>(
            [&](const auto Q) noexcept
            {
                accumulateDirection<Q>(moments, x, y, z, state, temperature, flux, step);
            });

        const real_t inverseDensity = 1 / state.rho;
        constexpr_for<0, 3>(
            [&](const auto D) noexcept
            {
                state.v[D] *= inverseDensity;
            });
        constexpr_for<0, 6>(
            [&](const auto D) noexcept
            {
                state.p[D] *= inverseDensity;
            });
    }

    state.p[0] -= state.v[0] * state.v[0];
    state.p[1] -= state.v[1] * state.v[1];
    state.p[2] -= state.v[2] * state.v[2];
    state.p[3] -= state.v[0] * state.v[1];
    state.p[4] -= state.v[0] * state.v[2];
    state.p[5] -= state.v[1] * state.v[2];
    temperature += validation::scalarSource(x, z, step + 1) / 2;

    real_t acceleration[3];
    nodeAcceleration(temperature, x, y, z, step + 1, acceleration);

    const real_t ramp = topSpongeRamp(z);
    const real_t omegaShear = 1 / (TAU_SHEAR + ramp * (TAU_SHEAR_SPONGE_MAX - TAU_SHEAR));
    const real_t omegaBulk = 1 / (TAU_BULK + ramp * (TAU_BULK_SPONGE_MAX - TAU_BULK));
    const real_t omegaThermal = 1 / (TAU_THERMAL + ramp * (TAU_THERMAL_SPONGE_MAX - TAU_THERMAL));
    real_t velocity[3] = {
        state.v[0] + acceleration[0] / 2,
        state.v[1] + acceleration[1] / 2,
        state.v[2] + acceleration[2] / 2};

    state = collideHydro(state, acceleration, omegaShear, omegaBulk);
#if defined(RBC_VALIDATION_CASE) && RBC_VALIDATION_CASE == 2
    // V2 isolates the scalar equation: velocity is prescribed, not solved.
    validation::prescribedVelocity(z, step + 1, velocity);
    state.rho = 1;
    constexpr_for<0, 3>([&](const auto D) noexcept { state.v[D] = velocity[D]; });
    constexpr_for<0, 6>([&](const auto D) noexcept { state.p[D] = 0; });
#endif

    const natural_t index = global3(x, y, z);
    storeMoment(dbuffer, index, RHO, state.rho);
    constexpr_for<0, 3>(
        [&](const auto D) noexcept
        {
            flux[D] = (1 - omegaThermal) * flux[D] + omegaThermal * temperature * velocity[D] + THERMAL_SOURCE_VALUE * velocity[D];
            storeMoment(dbuffer, index, UX + D, state.v[D]);
            storeMoment(dbuffer, index, TUX + D, flux[D]);
        });
    constexpr_for<0, 6>(
        [&](const auto D) noexcept
        {
            storeMoment(dbuffer, index, PXX + D, state.p[D]);
        });
    temperature += THERMAL_SOURCE_VALUE;
    storeMoment(dbuffer, index, TEMP, temperature);

#if defined(CASE_RAYLEIGH_BENARD)
    if (z == 1 || z == NZ - 2)
    {
        const natural_t wallZ = z == 1 ? 0 : NZ - 1;
        writeNEEPlate(plateStorage(dbuffer), global3(x, y, wallZ), wallZ, state, temperature, flux, acceleration);
    }
#endif
}

__global__ void streamCollide(
    const real_t *__restrict__ moments,
    real_t *__restrict__ dbuffer,
    const natural_t step)
{
    const natural_t x = blockIdx.x * BLOCK_NX + threadIdx.x;
    const natural_t y = blockIdx.y * BLOCK_NY + threadIdx.y;
    const natural_t z = blockIdx.z * BLOCK_NZ + threadIdx.z;
    streamCollideNode(moments, dbuffer, x, y, z, step);
}

#if defined(CASE_JET)
__global__ void accumulateThermalTimeAverage(const real_t *m, double *ta, double *ua)
{
    const natural_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= CELLS)
    {
        return;
    }

    real_t a[3];
    acceleration(m[midx(i, TEMP)], i / STRIDE > 0 && i / STRIDE < NZ - 1, a);
    ta[i] += m[midx(i, TEMP)];
    ua[i] += m[midx(i, UZ)] - a[2] / 2;
}

__global__ void finalizeThermalTimeAverage(const double *ta, const double *ua, real_t *tm, real_t *um, uint64_t n)
{
    const natural_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < CELLS)
    {
        tm[i] = ta[i] / n;
        um[i] = ua[i] / n;
    }
}
#endif
