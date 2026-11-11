#pragma once
#include "momentStorage.cuh"
#include "bitmasks.cuh"
#include "validation/physics.cuh"

// All moments here are divided by rho, except thermal moments.
struct HydroState
{
    real_t rho, v[3], p[6]; // xx, yy, zz, xy, xz, yz
};

__device__ __host__ inline void acceleration(real_t temperature, bool bulk, real_t (&a)[3])
{
    const real_t b = ENABLE_BUOYANCY && bulk ? BETA_THERMAL * (temperature - T_REF) : 0;
    a[0] = b * THERMAL_GRAVITY_X;
    a[1] = b * THERMAL_GRAVITY_Y;
    a[2] = b * THERMAL_GRAVITY_Z;
}

// Every time-dependent force consumer must use the time of its moment buffer.
__device__ __host__ inline void nodeAcceleration(real_t temperature,natural_t x,natural_t y,
                                                natural_t z,double step,real_t (&a)[3])
{
    acceleration(temperature,z>0 && z<NZ-1,a);
#if defined(RBC_VALIDATION_CASE) && RBC_VALIDATION_CASE == 3
    if(z>0 && z<NZ-1) validation::channelAcceleration(z,step,a);
#endif
}

// Force-adjusted stress C2 = A2/rho - uu - S2/2 on a post-collision field.
// With v=u+a/2 this is P + aa/4. Its pre-collision counterpart has +S2/2.
__device__ __host__ inline real_t rrThird(const HydroState &s, const real_t (&a)[3], int i, int j, int k)
{
    const real_t u[3] = {s.v[0] - a[0]/2, s.v[1] - a[1]/2, s.v[2] - a[2]/2};
    const int ij = i == j ? i : (i+j == 1 ? 3 : i+j == 2 ? 4 : 5);
    const int ik = i == k ? i : (i+k == 1 ? 3 : i+k == 2 ? 4 : 5);
    const int jk = j == k ? j : (j+k == 1 ? 3 : j+k == 2 ? 4 : 5);
    return u[i]*u[j]*u[k] +
        u[i]*(s.p[jk] + a[j]*a[k]/4) +
        u[j]*(s.p[ik] + a[i]*a[k]/4) +
        u[k]*(s.p[ij] + a[i]*a[j]/4) +
        (a[i]*u[j]*u[k] + a[j]*u[i]*u[k] + a[k]*u[i]*u[j])/2;
}

template <natural_t Q, int Order = RR_ORDER>
__device__ __host__ inline real_t hydroPopulation(const HydroState &s, const real_t (&a)[3])
{
    constexpr real_t cx = VelocitySet::cx<Q>(), cy = VelocitySet::cy<Q>(), cz = VelocitySet::cz<Q>();
    real_t value = 1 + 3*(cx*s.v[0] + cy*s.v[1] + cz*s.v[2]) + real_t(4.5)*(
        VelocitySet::hxx<Q>()*(s.v[0]*s.v[0]+s.p[0]) +
        VelocitySet::hyy<Q>()*(s.v[1]*s.v[1]+s.p[1]) +
        VelocitySet::hzz<Q>()*(s.v[2]*s.v[2]+s.p[2]) +
        2*VelocitySet::hxy<Q>()*(s.v[0]*s.v[1]+s.p[3]) +
        2*VelocitySet::hxz<Q>()*(s.v[0]*s.v[2]+s.p[4]) +
        2*VelocitySet::hyz<Q>()*(s.v[1]*s.v[2]+s.p[5]));
    if constexpr (Order == 3)
    {
        // Hxxx = Hyyy = Hzzz = 0 on D3Q27. Six xxy permutations
        // have multiplicity 3; xyz has multiplicity 6 in H3:A3/6cs^6.
        value += real_t(13.5)*(
            VelocitySet::hxx<Q>()*cy*rrThird(s,a,0,0,1) +
            VelocitySet::hxx<Q>()*cz*rrThird(s,a,0,0,2) +
            VelocitySet::hyy<Q>()*cx*rrThird(s,a,1,1,0) +
            VelocitySet::hyy<Q>()*cz*rrThird(s,a,1,1,2) +
            VelocitySet::hzz<Q>()*cx*rrThird(s,a,2,2,0) +
            VelocitySet::hzz<Q>()*cy*rrThird(s,a,2,2,1)) +
            27*cx*cy*cz*rrThird(s,a,0,1,2);
    }
    return VelocitySet::w<Q>() * s.rho * value;
}

// The passive-temperature lattice retains its established first-order equilibrium.
// Its non-equilibrium flux is reconstructed explicitly, without persistent g_i.
template <natural_t Q>
__device__ __host__ inline real_t thermalPopulation(real_t t, const real_t (&j)[3])
{
    return ThermalVelocitySet::w<Q>() * (t + 3*(ThermalVelocitySet::cx<Q>()*j[0] +
        ThermalVelocitySet::cy<Q>()*j[1] + ThermalVelocitySet::cz<Q>()*j[2]));
}

template <typename Input>
__device__ inline HydroState loadHydro(Input m, natural_t idx)
{
    HydroState s;
    s.rho = loadMoment(m,idx,RHO);
#pragma unroll
    for (int d=0; d<3; ++d) s.v[d] = loadMoment(m,idx,UX+d);
#pragma unroll
    for (int d=0; d<6; ++d) s.p[d] = loadMoment(m,idx,PXX+d);
    return s;
}

// q is pre-collision population velocity. Equivalent to Guo's time-centered
// forcing about u=q+a/2, including the quadratic force term after shifting to v.
__device__ __host__ inline HydroState collideHydro(HydroState pre, const real_t (&a)[3],
                                                  real_t ws, real_t wb)
{
    const real_t trace = (pre.p[0]+pre.p[1]+pre.p[2])/3;
    const real_t atrace = (a[0]*a[0]+a[1]*a[1]+a[2]*a[2])/12;
#pragma unroll
    for (int d=0; d<6; ++d)
    {
        const int i = d<3 ? d : d==3 || d==4 ? 0 : 1;
        const int j = d<3 ? d : d==3 ? 1 : 2;
        const real_t tr = d<3 ? trace : 0;
        const real_t at = d<3 ? atrace : 0;
        pre.p[d] = (1-ws)*(pre.p[d]-tr)+(1-wb)*tr -
                    ws*(a[i]*a[j]/4-at)-wb*at;
    }
#pragma unroll
    for (int d=0; d<3; ++d) pre.v[d] += a[d];
    return pre;
}
