#pragma once

#include "deviceFunctions.cuh"
#include "boundary/deviceRuntime.cuh"

template <natural_t Q>
__device__ static __forceinline__ void accumulateDirection(
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

    const real_t sourceTemperature = loadMoment(moments, src, TEMP);

    real_t cuT = static_cast<real_t>(0);

    if constexpr (cx != 0)
    {
        cuT += static_cast<real_t>(cx) * loadMoment(moments, src, TUX);
    }

    if constexpr (cy != 0)
    {
        cuT += static_cast<real_t>(cy) * loadMoment(moments, src, TUY);
    }

    if constexpr (cz != 0)
    {
        cuT += static_cast<real_t>(cz) * loadMoment(moments, src, TUZ);
    }

    const real_t gi = VelocitySet::w<Q>() * (sourceTemperature + cuT);

    temp += gi;

    if constexpr (cx != 0)
    {
        tux += gi * static_cast<real_t>(cx);
    }

    if constexpr (cy != 0)
    {
        tuy += gi * static_cast<real_t>(cy);
    }

    if constexpr (cz != 0)
    {
        tuz += gi * static_cast<real_t>(cz);
    }
}

__global__ void streamCollide(
    const real_t *__restrict__ moments,
    real_t *__restrict__ dbuffer,
    const natural_t step)
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
        copyOutflowMomentsFromLastPlane(moments, dbuffer, x, y, z, idx);
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
    real_t temp = static_cast<real_t>(0);
    real_t tux = static_cast<real_t>(0);
    real_t tuy = static_cast<real_t>(0);
    real_t tuz = static_cast<real_t>(0);

    if (nodeType != BULK)
    {
        dispatchIRBCBoundary(
            moments,
            x, y, z,
            step,
            nodeType,
            rho, ux, uy, uz,
            mxx, myy, mzz, mxy, mxz, myz,
            temp, tux, tuy, tuz);
    }
    else
    {
        constexpr_for<0, VelocitySet::Q()>(
            [&](const auto Q) noexcept
            {
                accumulateDirection<Q>(
                    moments,
                    x, y, z,
                    rho, ux, uy, uz,
                    mxx, myy, mzz, mxy, mxz, myz,
                    temp, tux, tuy, tuz);
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
    }

    const real_t invRho = static_cast<real_t>(1) / rho;

    ux *= VelocitySet::scaleI();
    uy *= VelocitySet::scaleI();
    uz *= VelocitySet::scaleI();
    mxx *= VelocitySet::scaleII();
    myy *= VelocitySet::scaleII();
    mzz *= VelocitySet::scaleII();
    mxy *= VelocitySet::scaleIJ();
    mxz *= VelocitySet::scaleIJ();
    myz *= VelocitySet::scaleIJ();
    tux *= VelocitySet::scaleI();
    tuy *= VelocitySet::scaleI();
    tuz *= VelocitySet::scaleI();

    const real_t buoyancy = (nodeType == BULK) ? BETA_THERMAL * THERMAL_FORCE_SCALE * (temp - T_REF) : static_cast<real_t>(0);
    const real_t forceX = buoyancy * THERMAL_GRAVITY_X;
    const real_t forceY = buoyancy * THERMAL_GRAVITY_Y;
    const real_t forceZ = buoyancy * THERMAL_GRAVITY_Z;

    const real_t uxEq = ux + static_cast<real_t>(1.5) * invRho * forceX;
    const real_t uyEq = uy + static_cast<real_t>(1.5) * invRho * forceY;
    const real_t uzEq = uz + static_cast<real_t>(1.5) * invRho * forceZ;

    const real_t trace = mxx + myy + mzz;
    const real_t traceEq = static_cast<real_t>(0.5) * (uxEq * uxEq + uyEq * uyEq + uzEq * uzEq);
    const real_t traceThird = VelocitySet::cs2() * trace;
    const real_t traceEqThird = VelocitySet::cs2() * traceEq;
    const real_t tracePostThird = VelocitySet::cs2() * (T_OMEGA_BULK * trace + OMEGA_BULK * traceEq);

    const real_t sourceScale = TT_OMEGA_SHEAR * VelocitySet::scaleI() * invRho;

    dbuffer[midx(idx, RHO)] = rho;
    dbuffer[midx(idx, UX)] = ux + VelocitySet::scaleI() * invRho * forceX;
    dbuffer[midx(idx, UY)] = uy + VelocitySet::scaleI() * invRho * forceY;
    dbuffer[midx(idx, UZ)] = uz + VelocitySet::scaleI() * invRho * forceZ;
    dbuffer[midx(idx, MXX)] = T_OMEGA_SHEAR * (mxx - traceThird) + OMEGA_SHEAR * (static_cast<real_t>(0.5) * uxEq * uxEq - traceEqThird) + tracePostThird + sourceScale * forceX * uxEq;
    dbuffer[midx(idx, MYY)] = T_OMEGA_SHEAR * (myy - traceThird) + OMEGA_SHEAR * (static_cast<real_t>(0.5) * uyEq * uyEq - traceEqThird) + tracePostThird + sourceScale * forceY * uyEq;
    dbuffer[midx(idx, MZZ)] = T_OMEGA_SHEAR * (mzz - traceThird) + OMEGA_SHEAR * (static_cast<real_t>(0.5) * uzEq * uzEq - traceEqThird) + tracePostThird + sourceScale * forceZ * uzEq;
    dbuffer[midx(idx, MXY)] = T_OMEGA_SHEAR * mxy + OMEGA_SHEAR * uxEq * uyEq + sourceScale * (forceX * uyEq + forceY * uxEq);
    dbuffer[midx(idx, MXZ)] = T_OMEGA_SHEAR * mxz + OMEGA_SHEAR * uxEq * uzEq + sourceScale * (forceX * uzEq + forceZ * uxEq);
    dbuffer[midx(idx, MYZ)] = T_OMEGA_SHEAR * myz + OMEGA_SHEAR * uyEq * uzEq + sourceScale * (forceY * uzEq + forceZ * uyEq);
    dbuffer[midx(idx, TEMP)] = temp + THERMAL_SOURCE_VALUE;
    dbuffer[midx(idx, TUX)] = T_OMEGA_THERMAL * tux + OMEGA_THERMAL * temp * uxEq + THERMAL_SOURCE_VALUE * uxEq;
    dbuffer[midx(idx, TUY)] = T_OMEGA_THERMAL * tuy + OMEGA_THERMAL * temp * uyEq + THERMAL_SOURCE_VALUE * uyEq;
    dbuffer[midx(idx, TUZ)] = T_OMEGA_THERMAL * tuz + OMEGA_THERMAL * temp * uzEq + THERMAL_SOURCE_VALUE * uzEq;
}
