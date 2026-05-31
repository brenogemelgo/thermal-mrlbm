#pragma once

#ifdef D3Q19
#include "D3Q19.cuh"
#elif defined(D3Q27)
#include "D3Q27.cuh"
#endif
#include "D3Q7.cuh"

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

__device__ static __forceinline__ void setStoredHydroEquilibriumMoments(
    real_t *__restrict__ moments,
    const natural_t idx,
    const real_t rho,
    const real_t ux,
    const real_t uy,
    const real_t uz) noexcept
{
    const real_t uxCoeff = ux * VelocitySet::scaleI();
    const real_t uyCoeff = uy * VelocitySet::scaleI();
    const real_t uzCoeff = uz * VelocitySet::scaleI();

    moments[midx(idx, RHO)] = rho;
    moments[midx(idx, UX)] = uxCoeff;
    moments[midx(idx, UY)] = uyCoeff;
    moments[midx(idx, UZ)] = uzCoeff;
    moments[midx(idx, MXX)] = static_cast<real_t>(0.5) * uxCoeff * uxCoeff;
    moments[midx(idx, MYY)] = static_cast<real_t>(0.5) * uyCoeff * uyCoeff;
    moments[midx(idx, MZZ)] = static_cast<real_t>(0.5) * uzCoeff * uzCoeff;
    moments[midx(idx, MXY)] = uxCoeff * uyCoeff;
    moments[midx(idx, MXZ)] = uxCoeff * uzCoeff;
    moments[midx(idx, MYZ)] = uyCoeff * uzCoeff;
}

__device__ static __forceinline__ void setStoredHydroMoments(
    real_t *__restrict__ moments,
    const natural_t idx,
    const real_t rho,
    const real_t ux,
    const real_t uy,
    const real_t uz,
    const real_t mxx,
    const real_t myy,
    const real_t mzz,
    const real_t mxy,
    const real_t mxz,
    const real_t myz) noexcept
{
    moments[midx(idx, RHO)] = rho;
    moments[midx(idx, UX)] = ux * VelocitySet::scaleI();
    moments[midx(idx, UY)] = uy * VelocitySet::scaleI();
    moments[midx(idx, UZ)] = uz * VelocitySet::scaleI();
    moments[midx(idx, MXX)] = mxx * VelocitySet::scaleII();
    moments[midx(idx, MYY)] = myy * VelocitySet::scaleII();
    moments[midx(idx, MZZ)] = mzz * VelocitySet::scaleII();
    moments[midx(idx, MXY)] = mxy * VelocitySet::scaleIJ();
    moments[midx(idx, MXZ)] = mxz * VelocitySet::scaleIJ();
    moments[midx(idx, MYZ)] = myz * VelocitySet::scaleIJ();
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

__device__ static __forceinline__ void setJetInletMoments(
    real_t *__restrict__ moments,
    const natural_t idx) noexcept
{
    setStoredHydroEquilibriumMoments(
        moments,
        idx,
        static_cast<real_t>(1),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        U_CHAR);
    setThermalEquilibriumMoments(
        moments,
        idx,
        T_HOT,
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        U_CHAR);
}

__device__ static __forceinline__ void setJetAmbientMoments(
    real_t *__restrict__ moments,
    const natural_t idx) noexcept
{
    setStoredHydroEquilibriumMoments(
        moments,
        idx,
        static_cast<real_t>(1),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0));
    setThermalEquilibriumMoments(
        moments,
        idx,
        T_COLD,
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0));
}

__device__ static __forceinline__ void setJetBedThermalMoments(
    const real_t *__restrict__ sourceMoments,
    real_t *__restrict__ targetMoments,
    const natural_t x,
    const natural_t y,
    const natural_t idx) noexcept
{
    const natural_t src = global3(x, y, 1);
    const real_t temperature = loadMoment(sourceMoments, src, TEMP);
    setThermalEquilibriumMoments(
        targetMoments,
        idx,
        temperature,
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0));
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
