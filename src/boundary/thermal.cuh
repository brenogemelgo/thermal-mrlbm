#pragma once

#include "../D3Q7.cuh"
#include "helpers.cuh"

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
        JET_UX,
        JET_UY,
        JET_UZ);
    setThermalEquilibriumMoments(
        moments,
        idx,
        JET_TEMPERATURE,
        JET_UX,
        JET_UY,
        JET_UZ);
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
        AMBIENT_TEMPERATURE,
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
    // The subsea bed is a solid wall. Its scalar value uses a zero-normal-gradient
    // copy from the adjacent fluid plane, with no wall advection.
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
