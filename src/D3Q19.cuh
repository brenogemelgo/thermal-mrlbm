#pragma once

#include "constants.cuh"

#ifdef D3Q19

struct VelocitySet
{
    __device__ __host__ [[nodiscard]] static __forceinline__ consteval natural_t Q() noexcept
    {
        return 27;
    }

    template <natural_t dir>
    __device__ __host__ [[nodiscard]] static __forceinline__ consteval int cx() noexcept
    {
        if constexpr (dir == 1 || dir == 7 || dir == 9 || dir == 13 || dir == 15)
        {
            return 1;
        }
        else if constexpr (dir == 2 || dir == 8 || dir == 10 || dir == 14 || dir == 16)
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
        if constexpr (dir == 3 || dir == 7 || dir == 11 || dir == 14 || dir == 17)
        {
            return 1;
        }
        else if constexpr (dir == 4 || dir == 8 || dir == 12 || dir == 13 || dir == 18)
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
        if constexpr (dir == 5 || dir == 9 || dir == 11 || dir == 16 || dir == 18)
        {
            return 1;
        }
        else if constexpr (dir == 6 || dir == 10 || dir == 12 || dir == 15 || dir == 17)
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
            return static_cast<real_t>(static_cast<double>(1) / static_cast<double>(3));
        }
        else if constexpr (dir <= 6)
        {
            return static_cast<real_t>(static_cast<double>(1) / static_cast<double>(18));
        }
        else
        {
            return static_cast<real_t>(static_cast<double>(1) / static_cast<double>(36));
        }
    }

    __device__ __host__ [[nodiscard]] static __forceinline__ consteval real_t cs2() noexcept
    {
        return static_cast<real_t>(static_cast<double>(1.0) / static_cast<double>(3.0));
    }

    __device__ __host__ [[nodiscard]] static __forceinline__ consteval real_t as2() noexcept
    {
        return static_cast<real_t>(3.0);
    }

    __device__ __host__ [[nodiscard]] static __forceinline__ consteval real_t as4() noexcept
    {
        return as2() * as2();
    }

    template <natural_t dir>
    __device__ __host__ [[nodiscard]] static __forceinline__ consteval real_t hxx() noexcept
    {
        return static_cast<real_t>(cx<dir>() * cx<dir>()) - cs2();
    }

    template <natural_t dir>
    __device__ __host__ [[nodiscard]] static __forceinline__ consteval real_t hyy() noexcept
    {
        return static_cast<real_t>(cy<dir>() * cy<dir>()) - cs2();
    }

    template <natural_t dir>
    __device__ __host__ [[nodiscard]] static __forceinline__ consteval real_t hzz() noexcept
    {
        return static_cast<real_t>(cz<dir>() * cz<dir>()) - cs2();
    }

    template <natural_t dir>
    __device__ __host__ [[nodiscard]] static __forceinline__ consteval real_t hxy() noexcept
    {
        return static_cast<real_t>(cx<dir>() * cy<dir>());
    }

    template <natural_t dir>
    __device__ __host__ [[nodiscard]] static __forceinline__ consteval real_t hxz() noexcept
    {
        return static_cast<real_t>(cx<dir>() * cz<dir>());
    }

    template <natural_t dir>
    __device__ __host__ [[nodiscard]] static __forceinline__ consteval real_t hyz() noexcept
    {
        return static_cast<real_t>(cy<dir>() * cz<dir>());
    }

    __device__ __host__ [[nodiscard]] static __forceinline__ consteval real_t scaleI() noexcept
    {
        return as2();
    }

    __device__ __host__ [[nodiscard]] static __forceinline__ consteval real_t scaleII() noexcept
    {
        return static_cast<real_t>(0.5) * as2() * as2();
    }

    __device__ __host__ [[nodiscard]] static __forceinline__ consteval real_t scaleIJ() noexcept
    {
        return as2() * as2();
    }
};

#endif