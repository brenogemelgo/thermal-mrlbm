#pragma once

#include "../syntheticInflow.cuh"
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
    ubz = nodeTypeValue == INLET ? U_CHAR : static_cast<real_t>(0);
}

__device__ static __forceinline__ void boundaryVelocityRuntime(
    const unsigned int nodeType,
    real_t &ubx,
    real_t &uby,
    real_t &ubz) noexcept
{
    ubx = static_cast<real_t>(0);
    uby = static_cast<real_t>(0);
    ubz = (nodeType & INLET_FLAG) == INLET_FLAG ? U_CHAR : static_cast<real_t>(0);
}

// ===================================================================================================================== //

template <unsigned int nodeTypeValue>
__device__ static __forceinline__ void applyThermalBoundaryTyped(
    const real_t ux,
    const real_t uy,
    const real_t uz,
    real_t &temp,
    real_t &tux,
    real_t &tuy,
    real_t &tuz) noexcept
{
    (void)ux;
    (void)uy;
    (void)uz;

    temp = T_COLD;
    tux = static_cast<real_t>(0);
    tuy = static_cast<real_t>(0);
    tuz = static_cast<real_t>(0);
}

// ===================================================================================================================== //

__device__ static __forceinline__ real_t loadClampedTemperatureNeighbor(
    const real_t *__restrict__ moments,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    const int dx,
    const int dy,
    const int dz) noexcept
{
    int xx = static_cast<int>(x) + dx;
    int yy = static_cast<int>(y) + dy;
    int zz = static_cast<int>(z) + dz;

    if (xx < 0)
    {
        xx = 0;
    }
    else if (xx >= static_cast<int>(NX))
    {
        xx = static_cast<int>(NX - 1u);
    }

    if (yy < 0)
    {
        yy = 0;
    }
    else if (yy >= static_cast<int>(NY))
    {
        yy = static_cast<int>(NY - 1u);
    }

    if (zz < 0)
    {
        zz = 0;
    }
    else if (zz >= static_cast<int>(NZ))
    {
        zz = static_cast<int>(NZ - 1u);
    }

    return loadMoment(moments, global3(static_cast<natural_t>(xx), static_cast<natural_t>(yy), static_cast<natural_t>(zz)), TEMP);
}

__device__ static __forceinline__ void applyWallThermalBoundary(
    const real_t *__restrict__ moments,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    const unsigned int nodeType,
    real_t &temp,
    real_t &tux,
    real_t &tuy,
    real_t &tuz) noexcept
{
#if defined(CASE_CAVITY) || defined(CASE_SLOT)
    if ((nodeType & XMIN_FACE) == XMIN_FACE)
    {
        temp = T_HOT;
    }
    else if ((nodeType & XMAX_FACE) == XMAX_FACE)
    {
        temp = T_COLD;
    }
    else if ((nodeType & YMIN_FACE) == YMIN_FACE)
    {
        temp = loadClampedTemperatureNeighbor(moments, x, y, z, 0, 1, 0);
    }
    else if ((nodeType & YMAX_FACE) == YMAX_FACE)
    {
        temp = loadClampedTemperatureNeighbor(moments, x, y, z, 0, -1, 0);
    }
    else if ((nodeType & ZMIN_FACE) == ZMIN_FACE)
    {
        temp = loadClampedTemperatureNeighbor(moments, x, y, z, 0, 0, 1);
    }
    else if ((nodeType & ZMAX_FACE) == ZMAX_FACE)
    {
        temp = loadClampedTemperatureNeighbor(moments, x, y, z, 0, 0, -1);
    }
    else
    {
        temp = T_REF;
    }
#else
    temp = T_COLD;
#endif

    tux = static_cast<real_t>(0);
    tuy = static_cast<real_t>(0);
    tuz = static_cast<real_t>(0);
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

__device__ static __forceinline__ void applyIRBCBoundaryRuntime(
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
    const natural_t tableOffset = static_cast<natural_t>(nodeType) * IRBC_TABLE_STRIDE;

    real_t ubx;
    real_t uby;
    real_t ubz;
    boundaryVelocityRuntime(nodeType, ubx, uby, ubz);

    real_t rhs[IRBC_UNKNOWNS] = {
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0)};

    constexpr_for<0, VelocitySet::Q()>(
        [&](const auto Q) noexcept
        {
            if (!isMissingDirectionRuntime<Q>(nodeType))
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

    real_t solved[IRBC_UNKNOWNS] = {};

#pragma unroll
    for (natural_t row = 0; row < IRBC_UNKNOWNS; ++row)
    {
        real_t sum = static_cast<real_t>(0);
#pragma unroll
        for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
        {
            sum += IRBC_INVERSE[tableOffset + row * IRBC_UNKNOWNS + col] * rhs[col];
        }
        solved[row] = sum;
    }

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

template <natural_t Q>
__device__ [[nodiscard]] static __forceinline__ real_t outflowLowerPopulation(
    const real_t rho,
    const real_t ux,
    const real_t uy,
    const real_t uz) noexcept
{
    constexpr int cx = VelocitySet::cx<Q>();
    constexpr int cy = VelocitySet::cy<Q>();
    constexpr int cz = VelocitySet::cz<Q>();

    const real_t cu = ux * static_cast<real_t>(cx) +
                      uy * static_cast<real_t>(cy) +
                      uz * static_cast<real_t>(cz);

    const real_t meqH =
        static_cast<real_t>(0.5) * VelocitySet::as4() *
            (ux * ux * VelocitySet::hxx<Q>() +
             uy * uy * VelocitySet::hyy<Q>() +
             uz * uz * VelocitySet::hzz<Q>()) +
        VelocitySet::as4() *
            (ux * uy * VelocitySet::hxy<Q>() +
             ux * uz * VelocitySet::hxz<Q>() +
             uy * uz * VelocitySet::hyz<Q>());

    return VelocitySet::w<Q>() * rho *
           (static_cast<real_t>(1) + VelocitySet::as2() * cu + OMEGA_SHEAR * meqH);
}

template <natural_t Q>
__device__ static __forceinline__ void accumulateOutflowIRBCRhs(
    const real_t *__restrict__ moments,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    const real_t rho,
    const real_t ux,
    const real_t uy,
    const real_t uz,
    real_t (&knownMoments)[6]) noexcept
{
    real_t f;

    if constexpr (isMissingDirectionConst<OUTFLOW, Q>())
    {
        f = outflowLowerPopulation<Q>(rho, ux, uy, uz);
    }
    else
    {
        f = reconstructPopulation<Q>(moments, x, y, z);
    }

    knownMoments[0] += f * VelocitySet::hxx<Q>();
    knownMoments[1] += f * VelocitySet::hyy<Q>();
    knownMoments[2] += f * VelocitySet::hzz<Q>();
    knownMoments[3] += f * VelocitySet::hxy<Q>();
    knownMoments[4] += f * VelocitySet::hxz<Q>();
    knownMoments[5] += f * VelocitySet::hyz<Q>();
}

__device__ static __forceinline__ void applyOutflowNeumannIRBC(
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
    real_t &myz,
    real_t &temp,
    real_t &tux,
    real_t &tuy,
    real_t &tuz) noexcept
{
    (void)z;

    const natural_t src = global3(x, y, NZ - 2u);

    rho = loadMoment(moments, src, RHO);
    ux = loadMoment(moments, src, UX) / VelocitySet::scaleI();
    uy = loadMoment(moments, src, UY) / VelocitySet::scaleI();
    uz = loadMoment(moments, src, UZ) / VelocitySet::scaleI();

    temp = loadMoment(moments, src, TEMP);
    tux = loadMoment(moments, src, TUX) / VelocitySet::scaleI();
    tuy = loadMoment(moments, src, TUY) / VelocitySet::scaleI();
    tuz = loadMoment(moments, src, TUZ) / VelocitySet::scaleI();

    real_t knownMoments[6] = {
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0)};

    constexpr_for<0, VelocitySet::Q()>(
        [&](const auto Q) noexcept
        {
            accumulateOutflowIRBCRhs<Q>(
                moments,
                x, y, NZ - 1u,
                rho, ux, uy, uz,
                knownMoments);
        });

    real_t rhs[IRBC_OUTFLOW_UNKNOWNS] = {
        knownMoments[0] - knownMoments[2],
        knownMoments[1] - knownMoments[2],
        knownMoments[3],
        knownMoments[4],
        knownMoments[5],
        rho * (ux * ux + uy * uy + uz * uz)};

    real_t solved[IRBC_OUTFLOW_UNKNOWNS] = {};

#pragma unroll
    for (natural_t row = 0; row < IRBC_OUTFLOW_UNKNOWNS; ++row)
    {
        real_t sum = static_cast<real_t>(0);
#pragma unroll
        for (natural_t col = 0; col < IRBC_OUTFLOW_UNKNOWNS; ++col)
        {
            sum += IRBC_OUTFLOW_INVERSE[row * IRBC_OUTFLOW_UNKNOWNS + col] * rhs[col];
        }
        solved[row] = sum;
    }

    const real_t invRho = static_cast<real_t>(1) / rho;
    mxx = solved[0] * invRho;
    myy = solved[1] * invRho;
    mzz = solved[2] * invRho;
    mxy = solved[3] * invRho;
    mxz = solved[4] * invRho;
    myz = solved[5] * invRho;
}

__device__ static __forceinline__ void dispatchIRBCBoundary(
    const real_t *__restrict__ moments,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    const natural_t step,
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
    real_t &myz,
    real_t &temp,
    real_t &tux,
    real_t &tuy,
    real_t &tuz) noexcept
{
    if ((nodeType & OUTFLOW_FLAG) == OUTFLOW_FLAG)
    {
        applyOutflowNeumannIRBC(
            moments,
            x, y, z,
            rho, ux, uy, uz,
            mxx, myy, mzz, mxy, mxz, myz,
            temp, tux, tuy, tuz);
        return;
    }

    if ((nodeType & INLET_FLAG) == INLET_FLAG)
    {
        applyIRBCBoundaryRuntime(moments, x, y, z, nodeType, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        {
            const real_t profile = jetInletProfile(x, y);
            const real_t envelope = turbulentInflowEnvelope(profile);
            const real_t uzMean = smoothJetMeanUz(profile);
            const real_t inletTemperature = smoothJetTemperature(profile);

            real_t duX = static_cast<real_t>(0);
            real_t duY = static_cast<real_t>(0);
            real_t duZ = static_cast<real_t>(0);

            if constexpr (ENABLE_SYNTHETIC_TURBULENT_INFLOW)
            {
                syntheticTurbulentInflowVelocity(x, y, step, envelope, duX, duY, duZ);
            }

            ux = duX;
            uy = duY;
            uz = uzMean + duZ;

            if (uz < static_cast<real_t>(0))
            {
                uz = static_cast<real_t>(0);
            }

            temp = inletTemperature;
            tux = temp * ux;
            tuy = temp * uy;
            tuz = temp * uz;
        }
        return;
    }

    if ((nodeType & FACE_MASK) != 0u)
    {
        applyIRBCBoundaryRuntime(moments, x, y, z, nodeType, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        applyWallThermalBoundary(moments, x, y, z, nodeType, temp, tux, tuy, tuz);
        return;
    }

    __builtin_unreachable();
}
