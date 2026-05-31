#pragma once

#include "../D3Q7.cuh"
#include "helpers.cuh"

__device__ [[nodiscard]] static __forceinline__ bool isThermalHotLid(
    const natural_t z) noexcept
{
    return z == NZ - 1;
}

__device__ [[nodiscard]] static __forceinline__ bool isThermalColdWall(
    const natural_t z) noexcept
{
    return z == 0;
}

__device__ [[nodiscard]] static __forceinline__ bool isThermalBoundary(
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    return x == 0 || x == NX - 1 ||
           y == 0 || y == NY - 1 ||
           z == 0 || z == NZ - 1;
}

__device__ [[nodiscard]] static __forceinline__ bool isThermalDirichletWall(
    const natural_t z) noexcept
{
    return isThermalHotLid(z) || isThermalColdWall(z);
}

__device__ [[nodiscard]] static __forceinline__ real_t thermalBoundaryTemperature(
    const natural_t z) noexcept
{
    return isThermalHotLid(z) ? T_HOT : T_COLD;
}

__device__ static __forceinline__ void thermalWallVelocity(
    const natural_t z,
    real_t &ux,
    real_t &uy,
    real_t &uz) noexcept
{
    if (isThermalHotLid(z))
    {
        ux = U_CHAR;
        uy = static_cast<real_t>(0);
        uz = static_cast<real_t>(0);
    }
    else
    {
        ux = static_cast<real_t>(0);
        uy = static_cast<real_t>(0);
        uz = static_cast<real_t>(0);
    }
}

__device__ static __forceinline__ void nearestThermalInteriorNode(
    const natural_t x,
    const natural_t y,
    const natural_t z,
    natural_t &xi,
    natural_t &yi,
    natural_t &zi) noexcept
{
    xi = x;
    yi = y;
    zi = z;

    if constexpr (NX > 1)
    {
        if (x == 0)
        {
            xi = 1;
        }
        else if (x == NX - 1)
        {
            xi = NX - 2;
        }
    }

    if constexpr (NY > 1)
    {
        if (y == 0)
        {
            yi = 1;
        }
        else if (y == NY - 1)
        {
            yi = NY - 2;
        }
    }

    if constexpr (NZ > 1)
    {
        if (z == 0)
        {
            zi = 1;
        }
        else if (z == NZ - 1)
        {
            zi = NZ - 2;
        }
    }
}

__device__ static __forceinline__ void setThermalEquilibriumMoments(
    real_t *__restrict__ moments,
    const natural_t idx,
    const real_t temperature,
    const real_t ux,
    const real_t uy,
    const real_t uz) noexcept
{
    const real_t invCs2 = ThermalVelocitySet::as2();

    moments[midx(idx, TEMP)] = temperature;
    moments[midx(idx, TFX)] = temperature * ux * invCs2;
    moments[midx(idx, TFY)] = temperature * uy * invCs2;
    moments[midx(idx, TFZ)] = temperature * uz * invCs2;
}

__device__ [[nodiscard]] static __forceinline__ bool applyThermalBoundary(
    const real_t *__restrict__ moments,
    real_t *__restrict__ output,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    const natural_t idx) noexcept
{
    if (!isThermalBoundary(x, y, z))
    {
        return false;
    }

    real_t temperature;
    if (isThermalDirichletWall(z))
    {
        temperature = thermalBoundaryTemperature(z);
    }
    else
    {
        natural_t xi;
        natural_t yi;
        natural_t zi;
        nearestThermalInteriorNode(x, y, z, xi, yi, zi);
        temperature = loadMoment(moments, global3(xi, yi, zi), TEMP);
    }

    real_t uxWall;
    real_t uyWall;
    real_t uzWall;
    thermalWallVelocity(z, uxWall, uyWall, uzWall);
    setThermalEquilibriumMoments(output, idx, temperature, uxWall, uyWall, uzWall);
    return true;
}
