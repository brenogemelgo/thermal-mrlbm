#pragma once

#include "deviceFunctions.cuh"

__global__ void cavityInit(
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

    moments[midx(idx, RHO)] = static_cast<real_t>(1);
    moments[midx(idx, UX)] = static_cast<real_t>(0);
    moments[midx(idx, UY)] = static_cast<real_t>(0);
    moments[midx(idx, UZ)] = static_cast<real_t>(0);
    moments[midx(idx, MXX)] = static_cast<real_t>(0);
    moments[midx(idx, MYY)] = static_cast<real_t>(0);
    moments[midx(idx, MZZ)] = static_cast<real_t>(0);
    moments[midx(idx, MXY)] = static_cast<real_t>(0);
    moments[midx(idx, MXZ)] = static_cast<real_t>(0);
    moments[midx(idx, MYZ)] = static_cast<real_t>(0);

    for (natural_t k = 0; k < NUM_MOMENTS; ++k)
    {
        dbuffer[midx(idx, k)] = moments[midx(idx, k)];
    }
}
