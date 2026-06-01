#pragma once

#include "constants.cuh"

constexpr uint8_t BULK = 0u;
constexpr uint8_t BACK = 1u << 4;
constexpr uint8_t FRONT = 1u << 5;
constexpr uint8_t BACK_FACE = BACK;
constexpr uint8_t FRONT_FACE = FRONT;

constexpr uint8_t WALL = BACK_FACE;
constexpr uint8_t INLET = BACK_FACE | (1u << 6);
constexpr uint8_t OUTFLOW = FRONT_FACE | (1u << 7);

__device__ [[nodiscard]] static __forceinline__ bool isJetInlet(
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    if (z != 0)
    {
        return false;
    }

    const real_t cx0 = static_cast<real_t>(0.5) * static_cast<real_t>(NX - 1);
    const real_t cy0 = static_cast<real_t>(0.5) * static_cast<real_t>(NY - 1);

    const real_t dx = static_cast<real_t>(x) - cx0;
    const real_t dy = static_cast<real_t>(y) - cy0;

    const real_t r2 = dx * dx + dy * dy;
    constexpr real_t radius2 = RADIUS * RADIUS;

    return r2 <= radius2;
}

__device__ [[nodiscard]] static __forceinline__ bool isJetBackSolidWall(
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    return z == 0 && !isJetInlet(x, y, z);
}

__device__ [[nodiscard]] static __forceinline__ bool isJetOutflow(
    const natural_t,
    const natural_t,
    const natural_t z) noexcept
{
    return z == NZ - 1;
}

__device__ [[nodiscard]] static __forceinline__ bool isJetBulk(
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    return !isJetInlet(x, y, z) && !isJetBackSolidWall(x, y, z) && !isJetOutflow(x, y, z);
}

__device__ [[nodiscard]] static __forceinline__ uint8_t boundaryMask(
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    if (isJetInlet(x, y, z))
    {
        return INLET;
    }
    if (isJetBackSolidWall(x, y, z))
    {
        return WALL;
    }
    if (isJetOutflow(x, y, z))
    {
        return OUTFLOW;
    }

    return BULK;
}
