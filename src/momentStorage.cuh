#pragma once

#include "deviceFunctions.cuh"

__device__ __forceinline__ real_t &momentRef(real_t *moments, natural_t index, natural_t field)
{
    return moments[midx(index, field)];
}

__device__ __forceinline__ void storeMoment(real_t *moments, natural_t index, natural_t field, real_t value)
{
    moments[midx(index, field)] = value;
}

template <natural_t Q>
__device__ __forceinline__ const real_t *directionMoments(const real_t *moments)
{
    return moments;
}

__device__ __forceinline__ real_t *plateStorage(real_t *moments)
{
    return moments;
}
