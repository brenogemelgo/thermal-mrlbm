#pragma once

#include "constants.cuh"

constexpr uint8_t BULK = 0u;
constexpr uint8_t XMIN_FACE = 1u << 0;
constexpr uint8_t XMAX_FACE = 1u << 1;
constexpr uint8_t YMIN_FACE = 1u << 2;
constexpr uint8_t YMAX_FACE = 1u << 3;
constexpr uint8_t ZMIN_FACE = 1u << 4;
constexpr uint8_t ZMAX_FACE = 1u << 5;
constexpr uint8_t BACK = ZMIN_FACE;
constexpr uint8_t FRONT = ZMAX_FACE;
constexpr uint8_t BACK_FACE = ZMIN_FACE;
constexpr uint8_t FRONT_FACE = ZMAX_FACE;
constexpr uint8_t FACE_MASK = XMIN_FACE | XMAX_FACE | YMIN_FACE | YMAX_FACE | ZMIN_FACE | ZMAX_FACE;

constexpr uint8_t WALL = ZMIN_FACE;
constexpr uint8_t INLET_FLAG = 1u << 6;
constexpr uint8_t OUTFLOW_FLAG = 1u << 7;
constexpr uint8_t INLET = ZMIN_FACE | INLET_FLAG;
constexpr uint8_t OUTFLOW = ZMAX_FACE | OUTFLOW_FLAG;

__device__ __host__ [[nodiscard]] static __forceinline__ bool isJetInlet(
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    if (z != 0)
    {
        return false;
    }

    if constexpr (!ENABLE_INLET)
    {
        return false;
    }

    const real_t cx0 = static_cast<real_t>(0.5) * static_cast<real_t>(NX - 1);
    const real_t cy0 = static_cast<real_t>(0.5) * static_cast<real_t>(NY - 1);

    const real_t dx = static_cast<real_t>(x) - cx0;
    const real_t dy = static_cast<real_t>(y) - cy0;

    const real_t r2 = dx * dx + dy * dy;
    constexpr real_t inletRadius = ENABLE_SMOOTH_JET_PROFILE
                                       ? RADIUS + JET_INLET_HALO_THICKNESS_CELLS
                                       : RADIUS;
    constexpr real_t radius2 = inletRadius * inletRadius;

    return r2 <= radius2;
}

__device__ __host__ [[nodiscard]] static __forceinline__ uint8_t wallFaceMask(
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    uint8_t mask = 0u;

    if constexpr (!ENABLE_WALLS)
    {
        return mask;
    }

    if constexpr (!PERIODIC_X)
    {
        if (x == 0u)
        {
            mask |= XMIN_FACE;
        }
        if (x == NX - 1u)
        {
            mask |= XMAX_FACE;
        }
    }

    if constexpr (!PERIODIC_Y)
    {
        if (y == 0u)
        {
            mask |= YMIN_FACE;
        }
        if (y == NY - 1u)
        {
            mask |= YMAX_FACE;
        }
    }

    if constexpr (!PERIODIC_Z)
    {
        if (z == 0u)
        {
            mask |= ZMIN_FACE;
        }
        if (z == NZ - 1u)
        {
            mask |= ZMAX_FACE;
        }
    }

    return mask;
}

__device__ __host__ [[nodiscard]] static __forceinline__ bool isJetBackSolidWall(
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    if constexpr (ENABLE_INLET)
    {
        return z == 0 && !isJetInlet(x, y, z);
    }
    else
    {
        return wallFaceMask(x, y, z) != 0u;
    }
}

__device__ __host__ [[nodiscard]] static __forceinline__ bool isJetOutflow(
    const natural_t,
    const natural_t,
    const natural_t z) noexcept
{
    if constexpr (ENABLE_OUTFLOW)
    {
        return z == NZ - 1u;
    }
    else
    {
        return false;
    }
}

__device__ __host__ [[nodiscard]] static __forceinline__ bool isJetBulk(
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    return !isJetInlet(x, y, z) && !isJetBackSolidWall(x, y, z) && !isJetOutflow(x, y, z);
}

__device__ __host__ [[nodiscard]] static __forceinline__ uint8_t boundaryMask(
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    if constexpr (!ENABLE_INLET && !ENABLE_OUTFLOW && !ENABLE_WALLS)
    {
        return BULK;
    }

    if (isJetInlet(x, y, z))
    {
        return INLET;
    }
    if (isJetBackSolidWall(x, y, z))
    {
        if constexpr (ENABLE_INLET)
        {
            return WALL;
        }
        else
        {
            return wallFaceMask(x, y, z);
        }
    }
    if (isJetOutflow(x, y, z))
    {
        return OUTFLOW;
    }

    return BULK;
}
