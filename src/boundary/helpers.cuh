#pragma once

#include "../bitmasks.cuh"
#include "../deviceFunctions.cuh"

// ===================================================================================================================== //

constexpr natural_t IRBC_UNKNOWNS = 7;
constexpr natural_t IRBC_TABLE_STRIDE = IRBC_UNKNOWNS * IRBC_UNKNOWNS;
constexpr natural_t IRBC_TABLE_NODE_TYPES = 256;
constexpr natural_t IRBC_TABLE_SIZE = IRBC_TABLE_NODE_TYPES * IRBC_TABLE_STRIDE;

__device__ __constant__ real_t IRBC_INVERSE[IRBC_TABLE_SIZE];

constexpr natural_t IRBC_OUTFLOW_UNKNOWNS = 6;
constexpr natural_t IRBC_OUTFLOW_TABLE_STRIDE = IRBC_OUTFLOW_UNKNOWNS * IRBC_OUTFLOW_UNKNOWNS;

__device__ __constant__ real_t IRBC_OUTFLOW_INVERSE[IRBC_OUTFLOW_TABLE_STRIDE];

// ===================================================================================================================== //

template <unsigned int nodeTypeValue, natural_t dir>
__device__ __host__ [[nodiscard]] static inline constexpr bool isMissingDirectionConst() noexcept
{
    return (((nodeTypeValue & XMIN_FACE) == XMIN_FACE) && (VelocitySet::cx<dir>() > 0)) ||
           (((nodeTypeValue & XMAX_FACE) == XMAX_FACE) && (VelocitySet::cx<dir>() < 0)) ||
           (((nodeTypeValue & YMIN_FACE) == YMIN_FACE) && (VelocitySet::cy<dir>() > 0)) ||
           (((nodeTypeValue & YMAX_FACE) == YMAX_FACE) && (VelocitySet::cy<dir>() < 0)) ||
           (((nodeTypeValue & ZMIN_FACE) == ZMIN_FACE) && (VelocitySet::cz<dir>() > 0)) ||
           (((nodeTypeValue & ZMAX_FACE) == ZMAX_FACE) && (VelocitySet::cz<dir>() < 0));
}

template <natural_t dir>
__device__ __host__ [[nodiscard]] static inline bool isMissingDirectionRuntime(
    const unsigned int nodeType) noexcept
{
    return (((nodeType & XMIN_FACE) == XMIN_FACE) && (VelocitySet::cx<dir>() > 0)) ||
           (((nodeType & XMAX_FACE) == XMAX_FACE) && (VelocitySet::cx<dir>() < 0)) ||
           (((nodeType & YMIN_FACE) == YMIN_FACE) && (VelocitySet::cy<dir>() > 0)) ||
           (((nodeType & YMAX_FACE) == YMAX_FACE) && (VelocitySet::cy<dir>() < 0)) ||
           (((nodeType & ZMIN_FACE) == ZMIN_FACE) && (VelocitySet::cz<dir>() > 0)) ||
           (((nodeType & ZMAX_FACE) == ZMAX_FACE) && (VelocitySet::cz<dir>() < 0));
}
