#pragma once
#include "../reconstruction.cuh"
#if defined(CASE_JET)
#include "jetIRBC.cuh"
#endif

#if defined(CASE_RAYLEIGH_BENARD)
// Complete post-collision on-node NEE, after the bulk kernel in the same stream.
// f*_b = feq_b + (f*_1 - feq_1 - S_1/2), projected to second order.
// g*_b = geq_b + (g*_1 - geq_1). All directions use the same adjacent node.
// Reading freshly updated interior moments avoids a time-lagged wall condition.
__global__ void applyNEEPlates(real_t *m,natural_t step=0)
{
    const natural_t x = blockIdx.x*blockDim.x+threadIdx.x;
    const natural_t y = blockIdx.y*blockDim.y+threadIdx.y;
    const natural_t z = blockIdx.z == 0 ? 0 : NZ-1;
    if (x>=NX || y>=NY) return;
    const natural_t b = global3(x,y,z), n = global3(x,y,z==0 ? 1 : NZ-2);
    const HydroState s = loadHydro(m,n);
    const real_t t = loadMoment(m,n,TEMP);
    real_t a[3]; nodeAcceleration(t,x,y,z==0 ? 1 : NZ-2,step,a);
    m[midx(b,RHO)] = s.rho;
#pragma unroll
    for (int d=0; d<3; ++d)
    {
        m[midx(b,UX+d)] = 0;
        m[midx(b,TUX+d)] = loadMoment(m,n,TUX+d)-t*(s.v[d]-a[d]/2);
    }
#pragma unroll
    for (int d=0; d<6; ++d)
    {
        const int i=d<3 ? d : d==3 || d==4 ? 0 : 1;
        const int j=d<3 ? d : d==3 ? 1 : 2;
        m[midx(b,PXX+d)] = s.p[d]+a[i]*a[j]/4;
    }
    m[midx(b,TEMP)] = z==0 ? T_HOT : T_COLD;
}
#endif
