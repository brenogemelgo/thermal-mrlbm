#pragma once

#include "D3Q7.cuh"
#include "deviceFunctions.cuh"
#include "boundary/deviceRuntime.cuh"
#include "boundary/thermal.cuh"

__device__ static __forceinline__ void copyAllMoments(
    real_t *__restrict__ target,
    const real_t *__restrict__ source,
    const natural_t targetIdx,
    const natural_t sourceIdx) noexcept
{
    for (natural_t moment = 0; moment < NUM_MOMENTS; ++moment)
    {
        target[midx(targetIdx, moment)] = source[midx(sourceIdx, moment)];
    }
}

template <natural_t Q>
__device__ [[nodiscard]] static __forceinline__ real_t reconstructThermalPopulation(
    const real_t *__restrict__ moments,
    const natural_t src) noexcept
{
    const real_t temperature = loadMoment(moments, src, TEMP);

    real_t cu = static_cast<real_t>(0);

    if constexpr (ThermalVelocitySet::cx<Q>() != 0)
    {
        cu += static_cast<real_t>(ThermalVelocitySet::cx<Q>()) * loadMoment(moments, src, TFX);
    }

    if constexpr (ThermalVelocitySet::cy<Q>() != 0)
    {
        cu += static_cast<real_t>(ThermalVelocitySet::cy<Q>()) * loadMoment(moments, src, TFY);
    }

    if constexpr (ThermalVelocitySet::cz<Q>() != 0)
    {
        cu += static_cast<real_t>(ThermalVelocitySet::cz<Q>()) * loadMoment(moments, src, TFZ);
    }

    return ThermalVelocitySet::w<Q>() * (temperature + cu);
}

template <natural_t Q>
__device__ static __forceinline__ void accumulateThermalDirection(
    const real_t *__restrict__ moments,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    real_t &temperature,
    real_t &jx,
    real_t &jy,
    real_t &jz) noexcept
{
    constexpr int cx = ThermalVelocitySet::cx<Q>();
    constexpr int cy = ThermalVelocitySet::cy<Q>();
    constexpr int cz = ThermalVelocitySet::cz<Q>();

    const natural_t src = jetPullNeighborIndex(x, y, z, cx, cy, cz);
    const real_t gi = reconstructThermalPopulation<Q>(moments, src);

    temperature += gi;

    if constexpr (cx != 0)
    {
        jx += gi * static_cast<real_t>(cx);
    }

    if constexpr (cy != 0)
    {
        jy += gi * static_cast<real_t>(cy);
    }

    if constexpr (cz != 0)
    {
        jz += gi * static_cast<real_t>(cz);
    }
}

template <natural_t Q>
__device__ static __forceinline__ void accumulateBulkDirection(
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
    constexpr int cx = VelocitySet::cx<Q>();
    constexpr int cy = VelocitySet::cy<Q>();
    constexpr int cz = VelocitySet::cz<Q>();

    const natural_t src = jetPullNeighborIndex(x, y, z, cx, cy, cz);

    real_t cu = static_cast<real_t>(0);

    if constexpr (cx != 0)
    {
        cu += static_cast<real_t>(cx) * loadMoment(moments, src, UX);
    }

    if constexpr (cy != 0)
    {
        cu += static_cast<real_t>(cy) * loadMoment(moments, src, UY);
    }

    if constexpr (cz != 0)
    {
        cu += static_cast<real_t>(cz) * loadMoment(moments, src, UZ);
    }

    real_t mh = loadMoment(moments, src, MXX) * VelocitySet::hxx<Q>() +
                loadMoment(moments, src, MYY) * VelocitySet::hyy<Q>() +
                loadMoment(moments, src, MZZ) * VelocitySet::hzz<Q>();

    if constexpr (cx * cy != 0)
    {
        mh += loadMoment(moments, src, MXY) * VelocitySet::hxy<Q>();
    }

    if constexpr (cx * cz != 0)
    {
        mh += loadMoment(moments, src, MXZ) * VelocitySet::hxz<Q>();
    }

    if constexpr (cy * cz != 0)
    {
        mh += loadMoment(moments, src, MYZ) * VelocitySet::hyz<Q>();
    }

    const real_t wrho = VelocitySet::w<Q>() * loadMoment(moments, src, RHO);
    const real_t fi = __fmaf_rn(wrho, cu + mh, wrho);

    rho += fi;

    if constexpr (cx != 0)
    {
        ux += fi * static_cast<real_t>(cx);
        mxx += fi;
    }

    if constexpr (cy != 0)
    {
        uy += fi * static_cast<real_t>(cy);
        myy += fi;
    }

    if constexpr (cz != 0)
    {
        uz += fi * static_cast<real_t>(cz);
        mzz += fi;
    }

    if constexpr (cx * cy != 0)
    {
        mxy += fi * static_cast<real_t>(cx * cy);
    }

    if constexpr (cx * cz != 0)
    {
        mxz += fi * static_cast<real_t>(cx * cz);
    }

    if constexpr (cy * cz != 0)
    {
        myz += fi * static_cast<real_t>(cy * cz);
    }
}

__global__ void stream(
    const real_t *__restrict__ moments,
    real_t *__restrict__ dbuffer)
{
    const natural_t x = blockIdx.x * BLOCK_NX + threadIdx.x;
    const natural_t y = blockIdx.y * BLOCK_NY + threadIdx.y;
    const natural_t z = blockIdx.z * BLOCK_NZ + threadIdx.z;

    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    const natural_t idx = global3(x, y, z);
    const uint8_t nodeType = boundaryMask(x, y, z);

    if (nodeType == OUTFLOW)
    {
        copyAllMoments(dbuffer, moments, idx, global3(x, y, NZ - 2));

        return;
    }

    if (nodeType == INLET)
    {
        setJetInletMoments(dbuffer, idx);

        return;
    }

    real_t rho = static_cast<real_t>(0);
    real_t ux = static_cast<real_t>(0);
    real_t uy = static_cast<real_t>(0);
    real_t uz = static_cast<real_t>(0);
    real_t mxx = static_cast<real_t>(0);
    real_t myy = static_cast<real_t>(0);
    real_t mzz = static_cast<real_t>(0);
    real_t mxy = static_cast<real_t>(0);
    real_t mxz = static_cast<real_t>(0);
    real_t myz = static_cast<real_t>(0);

    if (nodeType == WALL)
    {
        dispatchIRBCBoundary(
            moments,
            x,
            y,
            z,
            nodeType,
            rho,
            ux,
            uy,
            uz,
            mxx,
            myy,
            mzz,
            mxy,
            mxz,
            myz);

        setStoredHydroMoments(
            dbuffer,
            idx,
            rho,
            ux,
            uy,
            uz,
            mxx,
            myy,
            mzz,
            mxy,
            mxz,
            myz);

        setJetBedThermalMoments(moments, dbuffer, x, y, idx);

        return;
    }

    constexpr_for<0, VelocitySet::Q()>(
        [&](const auto Q) noexcept
        {
            accumulateBulkDirection<Q>(
                moments,
                x,
                y,
                z,
                rho,
                ux,
                uy,
                uz,
                mxx,
                myy,
                mzz,
                mxy,
                mxz,
                myz);
        });

    const real_t invRho = static_cast<real_t>(1) / rho;

    ux *= invRho;
    uy *= invRho;
    uz *= invRho;

    mxx = __fmaf_rn(mxx, invRho, -VelocitySet::cs2());
    myy = __fmaf_rn(myy, invRho, -VelocitySet::cs2());
    mzz = __fmaf_rn(mzz, invRho, -VelocitySet::cs2());
    mxy *= invRho;
    mxz *= invRho;
    myz *= invRho;

    dbuffer[midx(idx, RHO)] = rho;
    dbuffer[midx(idx, UX)] = ux;
    dbuffer[midx(idx, UY)] = uy;
    dbuffer[midx(idx, UZ)] = uz;
    dbuffer[midx(idx, MXX)] = mxx;
    dbuffer[midx(idx, MYY)] = myy;
    dbuffer[midx(idx, MZZ)] = mzz;
    dbuffer[midx(idx, MXY)] = mxy;
    dbuffer[midx(idx, MXZ)] = mxz;
    dbuffer[midx(idx, MYZ)] = myz;

    real_t temperature = static_cast<real_t>(0);
    real_t jx = static_cast<real_t>(0);
    real_t jy = static_cast<real_t>(0);
    real_t jz = static_cast<real_t>(0);

    constexpr_for<0, ThermalVelocitySet::Q()>(
        [&](const auto Q) noexcept
        {
            accumulateThermalDirection<Q>(
                moments,
                x,
                y,
                z,
                temperature,
                jx,
                jy,
                jz);
        });

    dbuffer[midx(idx, TEMP)] = temperature;
    dbuffer[midx(idx, TFX)] = jx;
    dbuffer[midx(idx, TFY)] = jy;
    dbuffer[midx(idx, TFZ)] = jz;
}

__global__ void collide(
    real_t *__restrict__ moments,
    const real_t *__restrict__ dbuffer)
{
    const natural_t x = blockIdx.x * BLOCK_NX + threadIdx.x;
    const natural_t y = blockIdx.y * BLOCK_NY + threadIdx.y;
    const natural_t z = blockIdx.z * BLOCK_NZ + threadIdx.z;

    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    const natural_t idx = global3(x, y, z);
    const uint8_t nodeType = boundaryMask(x, y, z);

    if (nodeType != BULK)
    {
        copyAllMoments(moments, dbuffer, idx, idx);

        return;
    }

    real_t rho = dbuffer[midx(idx, RHO)];
    real_t ux = dbuffer[midx(idx, UX)];
    real_t uy = dbuffer[midx(idx, UY)];
    real_t uz = dbuffer[midx(idx, UZ)];
    real_t mxx = dbuffer[midx(idx, MXX)];
    real_t myy = dbuffer[midx(idx, MYY)];
    real_t mzz = dbuffer[midx(idx, MZZ)];
    real_t mxy = dbuffer[midx(idx, MXY)];
    real_t mxz = dbuffer[midx(idx, MXZ)];
    real_t myz = dbuffer[midx(idx, MYZ)];

    real_t temperature = static_cast<real_t>(0);
    real_t jx = static_cast<real_t>(0);
    real_t jy = static_cast<real_t>(0);
    real_t jz = static_cast<real_t>(0);

    temperature = dbuffer[midx(idx, TEMP)];
    jx = dbuffer[midx(idx, TFX)];
    jy = dbuffer[midx(idx, TFY)];
    jz = dbuffer[midx(idx, TFZ)];

    const real_t invRho = static_cast<real_t>(1) / rho;

    real_t forceX = static_cast<real_t>(0);
    real_t forceY = static_cast<real_t>(0);
    real_t forceZ = static_cast<real_t>(0);

    if (nodeType == BULK)
    {
        const real_t buoyancy = THERMAL_RHO0 * BETA_THERMAL * THERMAL_FORCE_SCALE * (temperature - T_REF);

        forceX += buoyancy * THERMAL_GRAVITY_X;
        forceY += buoyancy * THERMAL_GRAVITY_Y;
        forceZ += buoyancy * THERMAL_GRAVITY_Z;
    }

    const real_t omega = OMEGA;
    const real_t tOmega = T_OMEGA;
    const real_t omegaBulk = OMEGA_BULK;
    const real_t tOmegaBulk = T_OMEGA_BULK;

    const real_t ttOmega = static_cast<real_t>(1) - static_cast<real_t>(0.5) * omega;
    const real_t ttOmegaBulk = static_cast<real_t>(1) - static_cast<real_t>(0.5) * omegaBulk;

    ux *= VelocitySet::scaleI();
    uy *= VelocitySet::scaleI();
    uz *= VelocitySet::scaleI();

    mxx *= VelocitySet::scaleII();
    myy *= VelocitySet::scaleII();
    mzz *= VelocitySet::scaleII();
    mxy *= VelocitySet::scaleIJ();
    mxz *= VelocitySet::scaleIJ();
    myz *= VelocitySet::scaleIJ();

    const real_t uxEq = ux + static_cast<real_t>(0.5) * VelocitySet::scaleI() * invRho * forceX;
    const real_t uyEq = uy + static_cast<real_t>(0.5) * VelocitySet::scaleI() * invRho * forceY;
    const real_t uzEq = uz + static_cast<real_t>(0.5) * VelocitySet::scaleI() * invRho * forceZ;

    const real_t uxStore = ux + VelocitySet::scaleI() * invRho * forceX;
    const real_t uyStore = uy + VelocitySet::scaleI() * invRho * forceY;
    const real_t uzStore = uz + VelocitySet::scaleI() * invRho * forceZ;

    const real_t eqXX = static_cast<real_t>(0.5) * uxEq * uxEq;
    const real_t eqYY = static_cast<real_t>(0.5) * uyEq * uyEq;
    const real_t eqZZ = static_cast<real_t>(0.5) * uzEq * uzEq;

    const real_t eqXY = uxEq * uyEq;
    const real_t eqXZ = uxEq * uzEq;
    const real_t eqYZ = uyEq * uzEq;

    const real_t oneThird = static_cast<real_t>(1.0 / 3.0);

    const real_t trace = mxx + myy + mzz;
    const real_t traceEq = eqXX + eqYY + eqZZ;

    const real_t traceThird = oneThird * trace;
    const real_t traceEqThird = oneThird * traceEq;

    const real_t tracePostThird =
        oneThird * (tOmegaBulk * trace + omegaBulk * traceEq);

    real_t mxxPost =
        tOmega * (mxx - traceThird) +
        omega * (eqXX - traceEqThird) +
        tracePostThird;

    real_t myyPost =
        tOmega * (myy - traceThird) +
        omega * (eqYY - traceEqThird) +
        tracePostThird;

    real_t mzzPost =
        tOmega * (mzz - traceThird) +
        omega * (eqZZ - traceEqThird) +
        tracePostThird;

    real_t mxyPost = tOmega * mxy + omega * eqXY;
    real_t mxzPost = tOmega * mxz + omega * eqXZ;
    real_t myzPost = tOmega * myz + omega * eqYZ;

    const real_t sxx = VelocitySet::scaleI() * invRho * forceX * uxEq;
    const real_t syy = VelocitySet::scaleI() * invRho * forceY * uyEq;
    const real_t szz = VelocitySet::scaleI() * invRho * forceZ * uzEq;

    const real_t sxy = VelocitySet::scaleI() * invRho * (forceX * uyEq + forceY * uxEq);
    const real_t sxz = VelocitySet::scaleI() * invRho * (forceX * uzEq + forceZ * uxEq);
    const real_t syz = VelocitySet::scaleI() * invRho * (forceY * uzEq + forceZ * uyEq);

    const real_t sourceTrace = sxx + syy + szz;
    const real_t sourceTraceThird = oneThird * sourceTrace;

    mxxPost += ttOmega * (sxx - sourceTraceThird) +
               ttOmegaBulk * sourceTraceThird;

    myyPost += ttOmega * (syy - sourceTraceThird) +
               ttOmegaBulk * sourceTraceThird;

    mzzPost += ttOmega * (szz - sourceTraceThird) +
               ttOmegaBulk * sourceTraceThird;

    mxyPost += ttOmega * sxy;
    mxzPost += ttOmega * sxz;
    myzPost += ttOmega * syz;

    moments[midx(idx, RHO)] = rho;
    moments[midx(idx, UX)] = uxStore;
    moments[midx(idx, UY)] = uyStore;
    moments[midx(idx, UZ)] = uzStore;
    moments[midx(idx, MXX)] = mxxPost;
    moments[midx(idx, MYY)] = myyPost;
    moments[midx(idx, MZZ)] = mzzPost;
    moments[midx(idx, MXY)] = mxyPost;
    moments[midx(idx, MXZ)] = mxzPost;
    moments[midx(idx, MYZ)] = myzPost;

    const real_t invHydroScaleI = static_cast<real_t>(1) / VelocitySet::scaleI();

    const real_t uxThermal = uxEq * invHydroScaleI;
    const real_t uyThermal = uyEq * invHydroScaleI;
    const real_t uzThermal = uzEq * invHydroScaleI;

    const real_t fluxOmega = static_cast<real_t>(1) - OMEGA_T;
    const real_t invThermalCs2 = ThermalVelocitySet::as2();

    const real_t temperaturePost = temperature + THERMAL_SOURCE_VALUE;

    const real_t qxPost =
        fluxOmega * jx +
        OMEGA_T * temperature * uxThermal +
        THERMAL_SOURCE_VALUE * uxThermal;

    const real_t qyPost =
        fluxOmega * jy +
        OMEGA_T * temperature * uyThermal +
        THERMAL_SOURCE_VALUE * uyThermal;

    const real_t qzPost =
        fluxOmega * jz +
        OMEGA_T * temperature * uzThermal +
        THERMAL_SOURCE_VALUE * uzThermal;

    moments[midx(idx, TEMP)] = temperaturePost;
    moments[midx(idx, TFX)] = qxPost * invThermalCs2;
    moments[midx(idx, TFY)] = qyPost * invThermalCs2;
    moments[midx(idx, TFZ)] = qzPost * invThermalCs2;
}

// =================================================================================================== //
