#pragma once
#include "reconstruction.cuh"
#include "boundary/deviceRuntime.cuh"

template <natural_t Q>
__device__ inline void accumulateDirection(const real_t *m, natural_t x, natural_t y, natural_t z,
                                           HydroState &s, real_t &t, real_t (&j)[3])
{
    constexpr int cx=VelocitySet::cx<Q>(), cy=VelocitySet::cy<Q>(), cz=VelocitySet::cz<Q>();
    const natural_t n=jetPullNeighborIndex(x,y,z,cx,cy,cz);
    const HydroState src=loadHydro(m,n);
    const natural_t sz=n/STRIDE;
    real_t a[3]; acceleration(loadMoment(m,n,TEMP),sz>0 && sz<NZ-1,a);
    const real_t f=hydroPopulation<Q>(src,a);
    s.rho += f;
    s.v[0] += cx*f; s.v[1] += cy*f; s.v[2] += cz*f;
    s.p[0] += VelocitySet::hxx<Q>()*f;
    s.p[1] += VelocitySet::hyy<Q>()*f;
    s.p[2] += VelocitySet::hzz<Q>()*f;
    s.p[3] += cx*cy*f; s.p[4] += cx*cz*f; s.p[5] += cy*cz*f;
    const real_t js[3]={loadMoment(m,n,TUX),loadMoment(m,n,TUY),loadMoment(m,n,TUZ)};
    const real_t g=thermalPopulation<Q>(loadMoment(m,n,TEMP),js);
    t += g; j[0] += cx*g; j[1] += cy*g; j[2] += cz*g;
}

__global__ void streamCollide(const real_t *__restrict__ moments, real_t *__restrict__ dbuffer,
                              const natural_t step)
{
    const natural_t x=blockIdx.x*BLOCK_NX+threadIdx.x;
    const natural_t y=blockIdx.y*BLOCK_NY+threadIdx.y;
    const natural_t z=blockIdx.z*BLOCK_NZ+threadIdx.z;
    if (x>=NX || y>=NY || z>=NZ) return;
    const natural_t idx=global3(x,y,z);
    const uint8_t type=boundaryMask(x,y,z);
#if defined(CASE_RAYLEIGH_BENARD)
    if (type!=BULK) return; // filled by applyNEEPlates after bulk collision
#endif
    HydroState s{};
    real_t t=0, j[3]={};
#if defined(CASE_JET)
    if (type!=BULK)
    {
        dispatchIRBCBoundary(moments,x,y,z,step,type,s.rho,s.v[0],s.v[1],s.v[2],
            s.p[0],s.p[1],s.p[2],s.p[3],s.p[4],s.p[5],t,j[0],j[1],j[2]);
        // Legacy jet dispatcher returns A2/rho, without the isotropic cs2.
    }
    else
#endif
    {
        constexpr_for<0,VelocitySet::Q()>([&](auto q) {
            accumulateDirection<q>(moments,x,y,z,s,t,j);
        });
        const real_t inv=1/s.rho;
#pragma unroll
        for (int d=0;d<3;++d) s.v[d]*=inv;
#pragma unroll
        for (int d=0;d<6;++d) s.p[d]*=inv;
    }
    s.p[0]-=s.v[0]*s.v[0]; s.p[1]-=s.v[1]*s.v[1]; s.p[2]-=s.v[2]*s.v[2];
    s.p[3]-=s.v[0]*s.v[1]; s.p[4]-=s.v[0]*s.v[2]; s.p[5]-=s.v[1]*s.v[2];
    real_t a[3]; acceleration(t,type==BULK,a);
    const real_t ramp=topSpongeRamp(z);
    const real_t ws=1/(TAU_SHEAR+ramp*(TAU_SHEAR_SPONGE_MAX-TAU_SHEAR));
    const real_t wb=1/(TAU_BULK+ramp*(TAU_BULK_SPONGE_MAX-TAU_BULK));
    const real_t wt=1/(TAU_THERMAL+ramp*(TAU_THERMAL_SPONGE_MAX-TAU_THERMAL));
    const real_t u[3]={s.v[0]+a[0]/2,s.v[1]+a[1]/2,s.v[2]+a[2]/2};
    s=collideHydro(s,a,ws,wb);
    dbuffer[midx(idx,RHO)]=s.rho;
#pragma unroll
    for(int d=0;d<3;++d)
    {
        dbuffer[midx(idx,UX+d)]=s.v[d];
        dbuffer[midx(idx,TUX+d)]=(1-wt)*j[d]+wt*t*u[d]+THERMAL_SOURCE_VALUE*u[d];
    }
#pragma unroll
    for(int d=0;d<6;++d) dbuffer[midx(idx,PXX+d)]=s.p[d];
    dbuffer[midx(idx,TEMP)]=t+THERMAL_SOURCE_VALUE;
}

#if defined(CASE_JET)
__global__ void accumulateThermalTimeAverage(const real_t *m, double *ta, double *ua)
{
    const natural_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=CELLS) return;
    real_t a[3]; acceleration(m[midx(i,TEMP)], i/STRIDE>0 && i/STRIDE<NZ-1,a);
    ta[i]+=m[midx(i,TEMP)]; ua[i]+=m[midx(i,UZ)]-a[2]/2;
}
__global__ void finalizeThermalTimeAverage(const double *ta,const double *ua,real_t *tm,real_t *um,uint64_t n)
{
    const natural_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<CELLS) {tm[i]=ta[i]/n; um[i]=ua[i]/n;}
}
#endif
