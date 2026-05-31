#pragma once

#include "D3Q7.cuh"
#include "deviceFunctions.cuh"
#include "boundary/deviceRuntime.cuh"

__device__ static __forceinline__ void copyAllMoments(
    real_t *__restrict__ target,
    const real_t *__restrict__ source,
    const natural_t targetIdx,
    const natural_t sourceIdx) noexcept
{
    constexpr_for<0, NUM_MOMENTS>(
        [&](const auto moment) noexcept
        {
            target[midx(targetIdx, moment)] = source[midx(sourceIdx, moment)];
        });
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
    const real_t fi = fma_rn(wrho, cu + mh, wrho);

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

    mxx = fma_rn(mxx, invRho, -VelocitySet::cs2());
    myy = fma_rn(myy, invRho, -VelocitySet::cs2());
    mzz = fma_rn(mzz, invRho, -VelocitySet::cs2());
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

    if (boundaryMask(x, y, z) != BULK)
    {
        copyAllMoments(moments, dbuffer, idx, idx);
        return;
    }

    const real_t rho = dbuffer[midx(idx, RHO)];
    const real_t invRho = static_cast<real_t>(1) / rho;

    const real_t scaleI = VelocitySet::scaleI();
    const real_t scaleII = VelocitySet::scaleII();
    const real_t scaleIJ = VelocitySet::scaleIJ();
    const real_t halfScaleI = static_cast<real_t>(0.5) * scaleI;
    const real_t invScaleI = static_cast<real_t>(1) / scaleI;
    const real_t oneThird = static_cast<real_t>(1.0 / 3.0);

    real_t ux = dbuffer[midx(idx, UX)] * scaleI;
    real_t uy = dbuffer[midx(idx, UY)] * scaleI;
    real_t uz = dbuffer[midx(idx, UZ)] * scaleI;

    real_t mxx = dbuffer[midx(idx, MXX)] * scaleII;
    real_t myy = dbuffer[midx(idx, MYY)] * scaleII;
    real_t mzz = dbuffer[midx(idx, MZZ)] * scaleII;
    real_t mxy = dbuffer[midx(idx, MXY)] * scaleIJ;
    real_t mxz = dbuffer[midx(idx, MXZ)] * scaleIJ;
    real_t myz = dbuffer[midx(idx, MYZ)] * scaleIJ;

    const real_t temperature = dbuffer[midx(idx, TEMP)];
    const real_t jx = dbuffer[midx(idx, TFX)];
    const real_t jy = dbuffer[midx(idx, TFY)];
    const real_t jz = dbuffer[midx(idx, TFZ)];

    const real_t buoyancy =
        BETA_THERMAL *
        THERMAL_FORCE_SCALE *
        (temperature - T_REF);

    const real_t forceX = buoyancy * THERMAL_GRAVITY_X;
    const real_t forceY = buoyancy * THERMAL_GRAVITY_Y;
    const real_t forceZ = buoyancy * THERMAL_GRAVITY_Z;

    const real_t uxEq = fma_rn(halfScaleI * invRho, forceX, ux);
    const real_t uyEq = fma_rn(halfScaleI * invRho, forceY, uy);
    const real_t uzEq = fma_rn(halfScaleI * invRho, forceZ, uz);

    const real_t uxStore = fma_rn(scaleI * invRho, forceX, ux);
    const real_t uyStore = fma_rn(scaleI * invRho, forceY, uy);
    const real_t uzStore = fma_rn(scaleI * invRho, forceZ, uz);

    const real_t eqXX = static_cast<real_t>(0.5) * uxEq * uxEq;
    const real_t eqYY = static_cast<real_t>(0.5) * uyEq * uyEq;
    const real_t eqZZ = static_cast<real_t>(0.5) * uzEq * uzEq;

    const real_t trace = mxx + myy + mzz;
    const real_t traceEq = eqXX + eqYY + eqZZ;

    const real_t traceThird = oneThird * trace;
    const real_t traceEqThird = oneThird * traceEq;
    const real_t tracePostThird =
        oneThird * (T_OMEGA_BULK * trace + OMEGA_BULK * traceEq);

    mxx =
        T_OMEGA_SHEAR * (mxx - traceThird) +
        OMEGA_SHEAR * (eqXX - traceEqThird) +
        tracePostThird;

    myy =
        T_OMEGA_SHEAR * (myy - traceThird) +
        OMEGA_SHEAR * (eqYY - traceEqThird) +
        tracePostThird;

    mzz =
        T_OMEGA_SHEAR * (mzz - traceThird) +
        OMEGA_SHEAR * (eqZZ - traceEqThird) +
        tracePostThird;

    mxy = fma_rn(OMEGA_SHEAR, uxEq * uyEq, T_OMEGA_SHEAR * mxy);
    mxz = fma_rn(OMEGA_SHEAR, uxEq * uzEq, T_OMEGA_SHEAR * mxz);
    myz = fma_rn(OMEGA_SHEAR, uyEq * uzEq, T_OMEGA_SHEAR * myz);

    const real_t ttOmegaShear =
        static_cast<real_t>(1) - static_cast<real_t>(0.5) * OMEGA_SHEAR;
    const real_t ttOmegaBulk =
        static_cast<real_t>(1) - static_cast<real_t>(0.5) * OMEGA_BULK;

    const real_t sxx = scaleI * invRho * forceX * uxEq;
    const real_t syy = scaleI * invRho * forceY * uyEq;
    const real_t szz = scaleI * invRho * forceZ * uzEq;

    const real_t sxy = scaleI * invRho * (forceX * uyEq + forceY * uxEq);
    const real_t sxz = scaleI * invRho * (forceX * uzEq + forceZ * uxEq);
    const real_t syz = scaleI * invRho * (forceY * uzEq + forceZ * uyEq);

    const real_t sourceTraceThird = oneThird * (sxx + syy + szz);

    mxx += ttOmegaShear * (sxx - sourceTraceThird) +
           ttOmegaBulk * sourceTraceThird;
    myy += ttOmegaShear * (syy - sourceTraceThird) +
           ttOmegaBulk * sourceTraceThird;
    mzz += ttOmegaShear * (szz - sourceTraceThird) +
           ttOmegaBulk * sourceTraceThird;
    mxy += ttOmegaShear * sxy;
    mxz += ttOmegaShear * sxz;
    myz += ttOmegaShear * syz;

    moments[midx(idx, RHO)] = rho;
    moments[midx(idx, UX)] = uxStore;
    moments[midx(idx, UY)] = uyStore;
    moments[midx(idx, UZ)] = uzStore;
    moments[midx(idx, MXX)] = mxx;
    moments[midx(idx, MYY)] = myy;
    moments[midx(idx, MZZ)] = mzz;
    moments[midx(idx, MXY)] = mxy;
    moments[midx(idx, MXZ)] = mxz;
    moments[midx(idx, MYZ)] = myz;

    const real_t uxThermal = uxEq * invScaleI;
    const real_t uyThermal = uyEq * invScaleI;
    const real_t uzThermal = uzEq * invScaleI;
    const real_t invThermalCs2 = ThermalVelocitySet::as2();
    const real_t temperaturePost = temperature + THERMAL_SOURCE_VALUE;

    moments[midx(idx, TEMP)] = temperaturePost;
    moments[midx(idx, TFX)] =
        invThermalCs2 *
        (T_OMEGA_THERMAL * jx +
         OMEGA_THERMAL * temperature * uxThermal +
         THERMAL_SOURCE_VALUE * uxThermal);
    moments[midx(idx, TFY)] =
        invThermalCs2 *
        (T_OMEGA_THERMAL * jy +
         OMEGA_THERMAL * temperature * uyThermal +
         THERMAL_SOURCE_VALUE * uyThermal);
    moments[midx(idx, TFZ)] =
        invThermalCs2 *
        (T_OMEGA_THERMAL * jz +
         OMEGA_THERMAL * temperature * uzThermal +
         THERMAL_SOURCE_VALUE * uzThermal);
}
