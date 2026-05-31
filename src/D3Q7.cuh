#pragma once

#include "constants.cuh"

struct ThermalVelocitySet
{
    __device__ __host__ [[nodiscard]] static __forceinline__ consteval natural_t Q() noexcept
    {
        return 7;
    }

    template <natural_t dir>
    __device__ __host__ [[nodiscard]] static __forceinline__ consteval int cx() noexcept
    {
        if constexpr (dir == 1)
        {
            return 1;
        }
        else if constexpr (dir == 2)
        {
            return -1;
        }
        else
        {
            return 0;
        }
    }

    template <natural_t dir>
    __device__ __host__ [[nodiscard]] static __forceinline__ consteval int cy() noexcept
    {
        if constexpr (dir == 3)
        {
            return 1;
        }
        else if constexpr (dir == 4)
        {
            return -1;
        }
        else
        {
            return 0;
        }
    }

    template <natural_t dir>
    __device__ __host__ [[nodiscard]] static __forceinline__ consteval int cz() noexcept
    {
        if constexpr (dir == 5)
        {
            return 1;
        }
        else if constexpr (dir == 6)
        {
            return -1;
        }
        else
        {
            return 0;
        }
    }

    template <natural_t dir>
    __device__ __host__ [[nodiscard]] static __forceinline__ consteval real_t w() noexcept
    {
        if constexpr (dir == 0)
        {
            return static_cast<real_t>(0.25);
        }
        else
        {
            return static_cast<real_t>(0.125);
        }
    }

    __device__ __host__ [[nodiscard]] static __forceinline__ consteval real_t cs2() noexcept
    {
        return THERMAL_CS2;
    }

    __device__ __host__ [[nodiscard]] static __forceinline__ consteval real_t as2() noexcept
    {
        return static_cast<real_t>(1) / cs2();
    }
};
