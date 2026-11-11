#pragma once

#include "reconstruction.cuh"
#include "deviceFunctions.cuh"
#include "syntheticInflow.cuh"

__device__ static __forceinline__ real_t caseSin(
    const real_t value) noexcept
{
    return sin(value);
}

__device__ static __forceinline__ void initialPrimitiveFields(
    const natural_t x,
    const natural_t y,
    const natural_t z,
    real_t &rho,
    real_t &ux,
    real_t &uy,
    real_t &uz,
    real_t &temperature) noexcept
{
    rho = ActiveCase::RHO0;
    ux = static_cast<real_t>(0);
    uy = static_cast<real_t>(0);
    uz = static_cast<real_t>(0);
    temperature = T_COLD;

#if defined(CASE_RAYLEIGH_BENARD)
    const real_t zNorm = ActiveCase::H > static_cast<real_t>(0)
                             ? static_cast<real_t>(z) / ActiveCase::H
                             : static_cast<real_t>(0);
    temperature = T_HOT - ActiveCase::DELTA_T * zNorm;

    if constexpr (!ActiveCase::CONDUCTION_ONLY)
    {
        const real_t xPhase = ActiveCase::TWO_PI * static_cast<real_t>(x) / static_cast<real_t>(NX);
        const real_t yPhase = ActiveCase::TWO_PI * static_cast<real_t>(y) / static_cast<real_t>(NY);
        const real_t zPhase = ActiveCase::PI * zNorm;

        const real_t perturb =
            caseSin(zPhase) *
            (caseSin(xPhase) + caseSin(yPhase) + static_cast<real_t>(0.5) * caseSin(xPhase + yPhase));

        temperature += ActiveCase::PERTURBATION_AMPLITUDE * perturb;
    }
#elif defined(CASE_JET)
    const bool inlet = isJetInlet(x, y, z);
    const real_t profile = inlet ? jetInletProfile(x, y) : static_cast<real_t>(0);
    const real_t envelope = turbulentInflowEnvelope(profile);

    uz = inlet ? smoothJetMeanUz(profile) : static_cast<real_t>(0);

    if (inlet)
    {
        real_t duX;
        real_t duY;
        real_t duZ;
        syntheticTurbulentInflowVelocity(x, y, 0u, envelope, duX, duY, duZ);

        ux = duX;
        uy = duY;
        uz += duZ;

        if (uz < static_cast<real_t>(0))
        {
            uz = static_cast<real_t>(0);
        }
    }

    temperature = inlet ? smoothJetTemperature(profile) : T_COLD;
#endif
}

template <typename Output>
__device__ __forceinline__ void initializeNode(
    Output moments,
    natural_t x,
    natural_t y,
    natural_t z)
{
    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    const natural_t idx = global3(x, y, z);
    real_t rho;
    real_t uxRaw;
    real_t uyRaw;
    real_t uzRaw;
    real_t temperature;
    initialPrimitiveFields(x, y, z, rho, uxRaw, uyRaw, uzRaw, temperature);

    real_t a[3];
#if defined(RBC_VALIDATION_CASE)
    real_t prescribed[3]; validation::prescribedVelocity(z,0,prescribed);
    uxRaw=prescribed[0]; uyRaw=prescribed[1]; uzRaw=prescribed[2];
#if RBC_VALIDATION_CASE == 2
    temperature+=ActiveCase::MODE_AMPLITUDE*sin(2*validation::PI*x/NX)*sin(validation::PI*z/(NZ-1));
#endif
#endif
    const real_t u[3] = {uxRaw, uyRaw, uzRaw};
    nodeAcceleration(temperature,x,y,z,0,a);
    momentRef(moments, idx, RHO) = rho;
    momentRef(moments, idx, TEMP) = temperature;
    for(int d=0; d<3; ++d)
    {
        momentRef(moments, idx, UX+d) = u[d]+a[d]/2;
        momentRef(moments, idx, TUX+d) = temperature*u[d];
    }
    for(int d=0; d<6; ++d)
    {
        const int i=d<3 ? d : d==3 || d==4 ? 0 : 1;
        const int j=d<3 ? d : d==3 ? 1 : 2;
        momentRef(moments, idx, PXX+d) = -a[i]*a[j]/4;
    }
#if defined(CASE_RAYLEIGH_BENARD)
    // Exact post-collision conductive flux for a linear temperature field.
    momentRef(moments, idx, TUZ) = (TAU_THERMAL-1)*ThermalVelocitySet::cs2()*ActiveCase::DELTA_T/ActiveCase::H;
#endif
#if defined(RBC_VALIDATION_CASE) && RBC_VALIDATION_CASE == 2
    const double h=NZ-1.,X=double(x)/NX,Z=double(z)/h;
    const double tx=ActiveCase::MODE_AMPLITUDE*2*validation::PI/NX*cos(2*validation::PI*X)*sin(validation::PI*Z);
    const double tz=(-1+ActiveCase::MODE_AMPLITUDE*validation::PI*sin(2*validation::PI*X)*cos(validation::PI*Z))/h;
    const double tdt=-THERMAL_DIFFUSIVITY/(h*h)*ActiveCase::MODE_AMPLITUDE*sin(2*validation::PI*X)*sin(validation::PI*Z);
    momentRef(moments, idx, TUX)=temperature*u[0]-(TAU_THERMAL-1)*(tx/3+tdt*u[0]);
    momentRef(moments, idx, TUZ)=-(TAU_THERMAL-1)*tz/3;
#endif
#if defined(RBC_VALIDATION_CASE) && RBC_VALIDATION_CASE == 3
    // Post-collision, force-corrected CE stress, not a persistent full moment.
    const double grad=ActiveCase::CHANNEL_RE*VISCOSITY/(ActiveCase::H*ActiveCase::H)*validation::PI*
        cos(validation::PI*z/ActiveCase::H);
    momentRef(moments, idx, PXZ)+=(1-OMEGA_SHEAR)*(-TAU_SHEAR*VelocitySet::cs2()*grad);
#endif

}

__global__ void initializeCase(real_t *moments, real_t *dbuffer)
{
    const natural_t x = threadIdx.x + BLOCK_NX * blockIdx.x;
    const natural_t y = threadIdx.y + BLOCK_NY * blockIdx.y;
    const natural_t z = threadIdx.z + BLOCK_NZ * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    initializeNode(moments, x, y, z);
    const natural_t index = global3(x, y, z);
    constexpr_for<0, NUM_MOMENTS>(
        [&](const auto Field) noexcept
        {
            dbuffer[midx(index, Field)] = moments[midx(index, Field)];
        });
}
