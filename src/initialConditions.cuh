#pragma once

#include "D3Q7.cuh"
#include "deviceFunctions.cuh"
#include "syntheticInflow.cuh"

__device__ static __forceinline__ real_t caseFourierSin(
    const real_t value) noexcept
{
    return sinf(value);
}

__device__ static __forceinline__ void initialPrimitiveFields(
    const natural_t x,
    const natural_t y,
    const natural_t z,
    real_t &rho,
    real_t &ux,
    real_t &uy,
    real_t &uz,
    real_t &temperature) noexcept
{
    rho = ActiveCase::RHO0;
    ux = static_cast<real_t>(0);
    uy = static_cast<real_t>(0);
    uz = static_cast<real_t>(0);
    temperature = T_COLD;

#if defined(CASE_FOURIER)
        const real_t phase =
            ActiveCase::KX * static_cast<real_t>(x) +
            ActiveCase::KY * static_cast<real_t>(y) +
            ActiveCase::KZ * static_cast<real_t>(z);

        ux = ActiveCase::UX0;
        uy = ActiveCase::UY0;
        uz = ActiveCase::UZ0;
        temperature = ActiveCase::T0 + ActiveCase::AMP_T * caseFourierSin(phase);
#elif defined(CASE_CAVITY)
        const real_t xNorm = NX > 1u
                                 ? static_cast<real_t>(x) / static_cast<real_t>(NX - 1u)
                                 : static_cast<real_t>(0);
        temperature = T_HOT + (T_COLD - T_HOT) * xNorm;
#elif defined(CASE_JET)
        const bool inlet = isJetInlet(x, y, z);
        const real_t profile = inlet ? jetInletProfile(x, y) : static_cast<real_t>(0);
        const real_t envelope = turbulentInflowEnvelope(profile);

        uz = inlet ? smoothJetMeanUz(profile) : static_cast<real_t>(0);

        if (inlet)
        {
            real_t duX;
            real_t duY;
            real_t duZ;
            syntheticTurbulentInflowVelocity(x, y, 0u, envelope, duX, duY, duZ);

            ux = duX;
            uy = duY;
            uz += duZ;

            if (uz < static_cast<real_t>(0))
            {
                uz = static_cast<real_t>(0);
            }
        }

        temperature = inlet ? smoothJetTemperature(profile) : T_COLD;
#endif
}

__global__ void initializeCase(
    real_t *moments,
    real_t *dbuffer)
{
    const natural_t x = threadIdx.x + BLOCK_NX * blockIdx.x;
    const natural_t y = threadIdx.y + BLOCK_NY * blockIdx.y;
    const natural_t z = threadIdx.z + BLOCK_NZ * blockIdx.z;

    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    const natural_t idx = global3(x, y, z);
    real_t rho;
    real_t uxRaw;
    real_t uyRaw;
    real_t uzRaw;
    real_t temperature;
    initialPrimitiveFields(x, y, z, rho, uxRaw, uyRaw, uzRaw, temperature);

    const real_t uxCoeff = uxRaw * VelocitySet::scaleI();
    const real_t uyCoeff = uyRaw * VelocitySet::scaleI();
    const real_t uzCoeff = uzRaw * VelocitySet::scaleI();
    const real_t invThermalCs2 = ThermalVelocitySet::as2();

    moments[midx(idx, RHO)] = rho;
    moments[midx(idx, UX)] = uxCoeff;
    moments[midx(idx, UY)] = uyCoeff;
    moments[midx(idx, UZ)] = uzCoeff;
    moments[midx(idx, MXX)] = static_cast<real_t>(0.5) * uxCoeff * uxCoeff;
    moments[midx(idx, MYY)] = static_cast<real_t>(0.5) * uyCoeff * uyCoeff;
    moments[midx(idx, MZZ)] = static_cast<real_t>(0.5) * uzCoeff * uzCoeff;
    moments[midx(idx, MXY)] = uxCoeff * uyCoeff;
    moments[midx(idx, MXZ)] = uxCoeff * uzCoeff;
    moments[midx(idx, MYZ)] = uyCoeff * uzCoeff;
    moments[midx(idx, TEMP)] = temperature;
    moments[midx(idx, TUX)] = temperature * uxRaw * invThermalCs2;
    moments[midx(idx, TUY)] = temperature * uyRaw * invThermalCs2;
    moments[midx(idx, TUZ)] = temperature * uzRaw * invThermalCs2;

    for (natural_t k = 0; k < NUM_MOMENTS; ++k)
    {
        dbuffer[midx(idx, k)] = moments[midx(idx, k)];
    }
}
