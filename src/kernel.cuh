#pragma once

#include "deviceFunctions.cuh"
#include "boundary/deviceRuntime.cuh"

template <natural_t Q>
__device__ static __forceinline__ void accumulateBulkDirection(
    const real_t *__restrict__ moments,
    const natural_t idx,
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
    constexpr int offset = cx + cy * static_cast<int>(NX) + cz * static_cast<int>(STRIDE);

    const natural_t src = idx - static_cast<natural_t>(offset);

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

__global__ void streamCollide(
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

    if (nodeType != BULK)
    {
        dispatchIRBCBoundary(moments, x, y, z, nodeType, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
    }
    else
    {
        constexpr_for<0, VelocitySet::Q()>(
            [&](const auto Q) noexcept
            {
                accumulateBulkDirection<Q>(moments, idx, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
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

    ux *= VelocitySet::scaleI();
    uy *= VelocitySet::scaleI();
    uz *= VelocitySet::scaleI();
    mxx *= VelocitySet::scaleII();
    myy *= VelocitySet::scaleII();
    mzz *= VelocitySet::scaleII();
    mxy *= VelocitySet::scaleIJ();
    mxz *= VelocitySet::scaleIJ();
    myz *= VelocitySet::scaleIJ();

    mxx = fma_rn(OMEGA_D2 * ux, ux, T_OMEGA * mxx);
    myy = fma_rn(OMEGA_D2 * uy, uy, T_OMEGA * myy);
    mzz = fma_rn(OMEGA_D2 * uz, uz, T_OMEGA * mzz);
    mxy = fma_rn(OMEGA * ux, uy, T_OMEGA * mxy);
    mxz = fma_rn(OMEGA * ux, uz, T_OMEGA * mxz);
    myz = fma_rn(OMEGA * uy, uz, T_OMEGA * myz);

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
}
