#pragma once

#include "D3Q7.cuh"
#include "deviceFunctions.cuh"
#include "syntheticInflow.cuh"

__global__ void jetInit(
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
    const bool inlet = isJetInlet(x, y, z);
    const real_t profile = inlet ? jetInletProfile(x, y) : static_cast<real_t>(0);
    const real_t envelope = turbulentInflowEnvelope(profile);

    real_t uxRaw = static_cast<real_t>(0);
    real_t uyRaw = static_cast<real_t>(0);
    real_t uzRaw = inlet ? smoothJetMeanUz(profile) : static_cast<real_t>(0);

    if (inlet)
    {
        real_t duX;
        real_t duY;
        real_t duZ;
        syntheticTurbulentInflowVelocity(x, y, 0u, envelope, duX, duY, duZ);

        uxRaw = duX;
        uyRaw = duY;
        uzRaw += duZ;

        if (uzRaw < static_cast<real_t>(0))
        {
            uzRaw = static_cast<real_t>(0);
        }
    }

    const real_t uxCoeff = uxRaw * VelocitySet::scaleI();
    const real_t uyCoeff = uyRaw * VelocitySet::scaleI();
    const real_t uzCoeff = uzRaw * VelocitySet::scaleI();
    const real_t temperature = inlet ? smoothJetTemperature(profile) : T_COLD;
    const real_t invThermalCs2 = ThermalVelocitySet::as2();

    moments[midx(idx, RHO)] = static_cast<real_t>(1);
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
