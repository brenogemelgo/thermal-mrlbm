#pragma once

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

    if (isJetInlet(x, y, z))
    {
        setJetInletMoments(moments, idx);
    }
    else
    {
        setJetAmbientMoments(moments, idx);
    }

    for (natural_t k = 0; k < NUM_MOMENTS; ++k)
    {
        dbuffer[midx(idx, k)] = moments[midx(idx, k)];
    }
}
