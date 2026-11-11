#pragma once

#include "../reconstruction.cuh"
#if defined(CASE_JET)
#include "jetIRBC.cuh"
#endif

#if defined(CASE_RAYLEIGH_BENARD)
// Complete post-collision on-node NEE from the adjacent interior donor.
// f*_b = feq_b + (f*_1 - feq_1 - S_1/2), projected to second order.
// g*_b = geq_b + (g*_1 - geq_1). All directions use the same adjacent node.
// The fused timestep calls this with register-resident post-collision moments.
template <typename Output>
__device__ __forceinline__ void writeNEEPlate(
    Output moments,
    natural_t boundary,
    natural_t z,
    const HydroState &state,
    real_t temperature,
    const real_t (&flux)[3],
    const real_t (&acceleration)[3])
{
    storeMoment(moments, boundary, RHO, state.rho);
    constexpr_for<0, 3>(
        [&](const auto D) noexcept
        {
            storeMoment(moments, boundary, UX + D, 0);
            storeMoment(moments, boundary, TUX + D, flux[D] - temperature * (state.v[D] - acceleration[D] / 2));
        });
    constexpr_for<0, 6>(
        [&](const auto D) noexcept
        {
            constexpr int i = D < 3 ? D : D == 3 || D == 4 ? 0 : 1;
            constexpr int j = D < 3 ? D : D == 3 ? 1 : 2;
            storeMoment(moments, boundary, PXX + D, state.p[D] + acceleration[i] * acceleration[j] / 4);
        });
    storeMoment(moments, boundary, TEMP, z == 0 ? T_HOT : T_COLD);
}

// Initialization and standalone boundary tests still need a separate pass:
// initializeCase writes every node, so concurrent plate writes would race.
__global__ void applyNEEPlates(real_t *moments, natural_t step = 0)
{
    const natural_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const natural_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const natural_t z = blockIdx.z == 0 ? 0 : NZ - 1;
    if (x >= NX || y >= NY)
    {
        return;
    }

    const natural_t donorZ = z == 0 ? 1 : NZ - 2;
    const natural_t donor = global3(x, y, donorZ);
    const HydroState state = loadHydro(moments, donor);
    const real_t temperature = loadMoment(moments, donor, TEMP);
    const real_t flux[3] = {
        loadMoment(moments, donor, TUX),
        loadMoment(moments, donor, TUY),
        loadMoment(moments, donor, TUZ)};

    real_t acceleration[3];
    nodeAcceleration(temperature, x, y, donorZ, step, acceleration);
    writeNEEPlate(moments, global3(x, y, z), z, state, temperature, flux, acceleration);
}
#endif
