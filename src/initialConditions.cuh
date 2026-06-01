#pragma once

#include "D3Q7.cuh"
#include "bitmasks.cuh"
#include "deviceFunctions.cuh"

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
    const real_t uxCoeff = static_cast<real_t>(0);
    const real_t uyCoeff = static_cast<real_t>(0);
    const real_t uzCoeff = inlet ? U_CHAR * VelocitySet::scaleI() : static_cast<real_t>(0);
    const real_t temperature = inlet ? T_HOT : T_COLD;
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
    moments[midx(idx, TUX)] = static_cast<real_t>(0);
    moments[midx(idx, TUY)] = static_cast<real_t>(0);
    moments[midx(idx, TUZ)] = inlet ? temperature * U_CHAR * invThermalCs2 : static_cast<real_t>(0);

    for (natural_t k = 0; k < NUM_MOMENTS; ++k)
    {
        dbuffer[midx(idx, k)] = moments[midx(idx, k)];
    }
}
