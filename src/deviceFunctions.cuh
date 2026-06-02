#pragma once

#include "D3Q7.cuh"
#include "D3Q19.cuh"
#include "D3Q27.cuh"

__device__ [[nodiscard]] static __forceinline__ natural_t global3(
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    return x + y * NX + z * STRIDE;
}

__device__ __host__ [[nodiscard]] static __forceinline__ natural_t midx(
    const natural_t idx,
    const natural_t moment) noexcept
{
    return idx + CELLS * moment;
}

__device__ [[nodiscard]] static __forceinline__ natural_t jetPullNeighborIndex(
    const natural_t x,
    const natural_t y,
    const natural_t z,
    const int cx,
    const int cy,
    const int cz) noexcept
{
    int xs = static_cast<int>(x) - cx;
    int ys = static_cast<int>(y) - cy;
    const int zs = static_cast<int>(z) - cz;

    xs = (xs + static_cast<int>(NX)) % static_cast<int>(NX);
    ys = (ys + static_cast<int>(NY)) % static_cast<int>(NY);

    return global3(
        static_cast<natural_t>(xs),
        static_cast<natural_t>(ys),
        static_cast<natural_t>(zs));
}

__device__ static __forceinline__ real_t fma_rn(real_t a, real_t b, real_t c)
{
    if constexpr (std::is_same_v<real_t, float>)
    {
        return __fmaf_rn(a, b, c);
    }
    else
    {
        return __fma_rn(a, b, c);
    }
}

__device__ [[nodiscard]] static __forceinline__ real_t loadMoment(
    const real_t *__restrict__ moments,
    const natural_t idx,
    const natural_t moment) noexcept
{
    return __ldg(moments + midx(idx, moment));
}

__device__ __host__ [[nodiscard]] static __forceinline__ real_t topSpongeRamp(
    const natural_t z) noexcept
{
    const real_t zn = static_cast<real_t>(z) * SPONGE_INV_NZ_M1;
    real_t s = (zn - SPONGE_Z_START) * SPONGE_INV_WIDTH;

    if (s < static_cast<real_t>(0))
    {
        s = static_cast<real_t>(0);
    }
    else if (s > static_cast<real_t>(1))
    {
        s = static_cast<real_t>(1);
    }

    return s * s * (static_cast<real_t>(3) - static_cast<real_t>(2) * s);
}

template <typename T, T v>
struct IntegralConstant
{
    static constexpr const T value = v;
    using value_type = T;
    using type = IntegralConstant;

    __device__ [[nodiscard]] inline consteval operator value_type() const noexcept
    {
        return value;
    }

    __device__ [[nodiscard]] inline consteval value_type operator()() const noexcept
    {
        return value;
    }
};

template <const natural_t Start, const natural_t End, typename F>
__device__ __forceinline__ constexpr void constexpr_for(F &&f) noexcept
{
    if constexpr (Start < End)
    {
        f(IntegralConstant<natural_t, Start>());
        if constexpr (Start + 1 < End)
        {
            constexpr_for<Start + 1, End>(std::forward<F>(f));
        }
    }
}
