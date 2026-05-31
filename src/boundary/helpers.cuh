#pragma once

#include "../bitmasks.cuh"
#include "../deviceFunctions.cuh"

// ===================================================================================================================== //

constexpr natural_t IRBC_UNKNOWNS = 7;
constexpr natural_t IRBC_TABLE_STRIDE = IRBC_UNKNOWNS * IRBC_UNKNOWNS;
constexpr natural_t IRBC_TABLE_SIZE = 64 * IRBC_TABLE_STRIDE;

__device__ __constant__ real_t IRBC_INVERSE[IRBC_TABLE_SIZE];

// ===================================================================================================================== //

template <unsigned int nodeTypeValue, natural_t dir>
__device__ __host__ [[nodiscard]] static inline constexpr bool isMissingDirectionConst() noexcept
{
    return (((nodeTypeValue & WEST) == WEST) && (VelocitySet::cx<dir>() > 0)) ||
           (((nodeTypeValue & EAST) == EAST) && (VelocitySet::cx<dir>() < 0)) ||
           (((nodeTypeValue & SOUTH) == SOUTH) && (VelocitySet::cy<dir>() > 0)) ||
           (((nodeTypeValue & NORTH) == NORTH) && (VelocitySet::cy<dir>() < 0)) ||
           (((nodeTypeValue & BACK) == BACK) && (VelocitySet::cz<dir>() > 0)) ||
           (((nodeTypeValue & FRONT) == FRONT) && (VelocitySet::cz<dir>() < 0));
}
