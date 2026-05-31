#pragma once

#include "helpers.cuh"

// ===================================================================================================================== //

template <natural_t Q>
__device__ [[nodiscard]] static __forceinline__ real_t reconstructPopulation(
    const real_t *__restrict__ moments,
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    constexpr int cx = VelocitySet::cx<Q>();
    constexpr int cy = VelocitySet::cy<Q>();
    constexpr int cz = VelocitySet::cz<Q>();

    const natural_t src = jetPullNeighborIndex(x, y, z, cx, cy, cz);

    const real_t cu = static_cast<real_t>(cx) * loadMoment(moments, src, UX) +
                      static_cast<real_t>(cy) * loadMoment(moments, src, UY) +
                      static_cast<real_t>(cz) * loadMoment(moments, src, UZ);

    const real_t mh = loadMoment(moments, src, MXX) * VelocitySet::hxx<Q>() +
                      loadMoment(moments, src, MYY) * VelocitySet::hyy<Q>() +
                      loadMoment(moments, src, MZZ) * VelocitySet::hzz<Q>() +
                      loadMoment(moments, src, MXY) * VelocitySet::hxy<Q>() +
                      loadMoment(moments, src, MXZ) * VelocitySet::hxz<Q>() +
                      loadMoment(moments, src, MYZ) * VelocitySet::hyz<Q>();

    const real_t wrho = VelocitySet::w<Q>() * loadMoment(moments, src, RHO);
    return fma_rn(wrho, cu + mh, wrho);
}

// ===================================================================================================================== //

template <unsigned int nodeTypeValue>
__device__ static __forceinline__ void boundaryVelocityConst(
    real_t &ubx,
    real_t &uby,
    real_t &ubz) noexcept
{
    ubx = static_cast<real_t>(0);
    uby = static_cast<real_t>(0);
    ubz = static_cast<real_t>(0);
}

// ===================================================================================================================== //

template <unsigned int nodeTypeValue>
__device__ static __forceinline__ void applyIRBCBoundaryTyped(
    const real_t *__restrict__ moments,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    real_t &rho,
    real_t &ux,
    real_t &uy,
    real_t &uz,
    real_t &mxx,
    real_t &myy,
    real_t &mzz,
    real_t &mxy,
    real_t &mxz,
    real_t &myz) noexcept
{
    constexpr natural_t tableOffset = static_cast<natural_t>(nodeTypeValue) * IRBC_TABLE_STRIDE;

    real_t ubx;
    real_t uby;
    real_t ubz;
    boundaryVelocityConst<nodeTypeValue>(ubx, uby, ubz);

    real_t rhs[IRBC_UNKNOWNS] = {static_cast<real_t>(0),
                                 static_cast<real_t>(0),
                                 static_cast<real_t>(0),
                                 static_cast<real_t>(0),
                                 static_cast<real_t>(0),
                                 static_cast<real_t>(0),
                                 static_cast<real_t>(0)};

    constexpr_for<0, VelocitySet::Q()>(
        [&](const auto Q) noexcept
        {
            if constexpr (!isMissingDirectionConst<nodeTypeValue, Q>())
            {
                const real_t f = reconstructPopulation<Q>(moments, x, y, z);

                rhs[0] += f;
                rhs[1] += f * (VelocitySet::hxx<Q>() - VelocitySet::hzz<Q>());
                rhs[2] += f * (VelocitySet::hyy<Q>() - VelocitySet::hzz<Q>());
                rhs[3] += f * VelocitySet::hxy<Q>();
                rhs[4] += f * VelocitySet::hxz<Q>();
                rhs[5] += f * VelocitySet::hyz<Q>();
            }
        });

    rhs[6] = static_cast<real_t>(0);

    real_t solved[IRBC_UNKNOWNS] = {static_cast<real_t>(0),
                                    static_cast<real_t>(0),
                                    static_cast<real_t>(0),
                                    static_cast<real_t>(0),
                                    static_cast<real_t>(0),
                                    static_cast<real_t>(0),
                                    static_cast<real_t>(0)};

    solved[0] = fma_rn(IRBC_INVERSE[tableOffset + 0], rhs[0], fma_rn(IRBC_INVERSE[tableOffset + 1], rhs[1], fma_rn(IRBC_INVERSE[tableOffset + 2], rhs[2], fma_rn(IRBC_INVERSE[tableOffset + 3], rhs[3], fma_rn(IRBC_INVERSE[tableOffset + 4], rhs[4], fma_rn(IRBC_INVERSE[tableOffset + 5], rhs[5], IRBC_INVERSE[tableOffset + 6] * rhs[6]))))));
    solved[1] = fma_rn(IRBC_INVERSE[tableOffset + 7], rhs[0], fma_rn(IRBC_INVERSE[tableOffset + 8], rhs[1], fma_rn(IRBC_INVERSE[tableOffset + 9], rhs[2], fma_rn(IRBC_INVERSE[tableOffset + 10], rhs[3], fma_rn(IRBC_INVERSE[tableOffset + 11], rhs[4], fma_rn(IRBC_INVERSE[tableOffset + 12], rhs[5], IRBC_INVERSE[tableOffset + 13] * rhs[6]))))));
    solved[2] = fma_rn(IRBC_INVERSE[tableOffset + 14], rhs[0], fma_rn(IRBC_INVERSE[tableOffset + 15], rhs[1], fma_rn(IRBC_INVERSE[tableOffset + 16], rhs[2], fma_rn(IRBC_INVERSE[tableOffset + 17], rhs[3], fma_rn(IRBC_INVERSE[tableOffset + 18], rhs[4], fma_rn(IRBC_INVERSE[tableOffset + 19], rhs[5], IRBC_INVERSE[tableOffset + 20] * rhs[6]))))));
    solved[3] = fma_rn(IRBC_INVERSE[tableOffset + 21], rhs[0], fma_rn(IRBC_INVERSE[tableOffset + 22], rhs[1], fma_rn(IRBC_INVERSE[tableOffset + 23], rhs[2], fma_rn(IRBC_INVERSE[tableOffset + 24], rhs[3], fma_rn(IRBC_INVERSE[tableOffset + 25], rhs[4], fma_rn(IRBC_INVERSE[tableOffset + 26], rhs[5], IRBC_INVERSE[tableOffset + 27] * rhs[6]))))));
    solved[4] = fma_rn(IRBC_INVERSE[tableOffset + 28], rhs[0], fma_rn(IRBC_INVERSE[tableOffset + 29], rhs[1], fma_rn(IRBC_INVERSE[tableOffset + 30], rhs[2], fma_rn(IRBC_INVERSE[tableOffset + 31], rhs[3], fma_rn(IRBC_INVERSE[tableOffset + 32], rhs[4], fma_rn(IRBC_INVERSE[tableOffset + 33], rhs[5], IRBC_INVERSE[tableOffset + 34] * rhs[6]))))));
    solved[5] = fma_rn(IRBC_INVERSE[tableOffset + 35], rhs[0], fma_rn(IRBC_INVERSE[tableOffset + 36], rhs[1], fma_rn(IRBC_INVERSE[tableOffset + 37], rhs[2], fma_rn(IRBC_INVERSE[tableOffset + 38], rhs[3], fma_rn(IRBC_INVERSE[tableOffset + 39], rhs[4], fma_rn(IRBC_INVERSE[tableOffset + 40], rhs[5], IRBC_INVERSE[tableOffset + 41] * rhs[6]))))));
    solved[6] = fma_rn(IRBC_INVERSE[tableOffset + 42], rhs[0], fma_rn(IRBC_INVERSE[tableOffset + 43], rhs[1], fma_rn(IRBC_INVERSE[tableOffset + 44], rhs[2], fma_rn(IRBC_INVERSE[tableOffset + 45], rhs[3], fma_rn(IRBC_INVERSE[tableOffset + 46], rhs[4], fma_rn(IRBC_INVERSE[tableOffset + 47], rhs[5], IRBC_INVERSE[tableOffset + 48] * rhs[6]))))));

    rho = solved[0];

    const real_t invRho = static_cast<real_t>(1) / rho;

    ux = ubx;
    uy = uby;
    uz = ubz;

    mxx = solved[1] * invRho;
    myy = solved[2] * invRho;
    mzz = solved[3] * invRho;
    mxy = solved[4] * invRho;
    mxz = solved[5] * invRho;
    myz = solved[6] * invRho;
}

__device__ static __forceinline__ void dispatchIRBCBoundary(
    const real_t *__restrict__ moments,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    const unsigned int nodeType,
    real_t &rho,
    real_t &ux,
    real_t &uy,
    real_t &uz,
    real_t &mxx,
    real_t &myy,
    real_t &mzz,
    real_t &mxy,
    real_t &mxz,
    real_t &myz) noexcept
{
    switch (nodeType)
    {
    case WALL:
        applyIRBCBoundaryTyped<BACK_FACE>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    default:
        __builtin_unreachable();
    }
}
