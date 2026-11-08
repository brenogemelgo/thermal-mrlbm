#pragma once
// Analytic *driving* only. Python references are independently implemented.
// This header depends on constants and is included after their declarations.
namespace validation
{
#if defined(RBC_VALIDATION_CASE)
constexpr int KIND = RBC_VALIDATION_CASE;
#else
constexpr int KIND = 0;
#endif
constexpr double PI = 3.1415926535897932384626433832795;

__device__ __host__ inline real_t scalarSource(natural_t x,natural_t z,double step)
{
#if defined(RBC_VALIDATION_CASE) && RBC_VALIDATION_CASE == 2
    if(z==0 || z==NZ-1) return 0;
    const double h=NZ-1., k=THERMAL_DIFFUSIVITY;
    const double X=double(x)/NX,Z=double(z)/h,s=k*step/(h*h);
    const double mode=double(ActiveCase::MODE_AMPLITUDE)*exp(-s)*sin(2*PI*X)*sin(PI*Z);
    const double tx=double(ActiveCase::MODE_AMPLITUDE)*exp(-s)*2*PI/NX*cos(2*PI*X)*sin(PI*Z);
    const double vel=ActiveCase::PECLET*k/h*sin(PI*Z)*cos(2*PI*s);
    const double lap=-((2*PI/NX)*(2*PI/NX)+(PI/h)*(PI/h))*mode;
    return static_cast<real_t>(-k/(h*h)*mode+vel*tx-k*lap);
#else
    return 0;
#endif
}
__device__ __host__ inline void prescribedVelocity(natural_t z,double step,real_t (&u)[3])
{
    u[0]=u[1]=u[2]=0;
#if defined(RBC_VALIDATION_CASE)
    const double h=NZ-1.,Z=double(z)/h;
    if constexpr (KIND==2)
    {
        const double s=double(THERMAL_DIFFUSIVITY)*step/(h*h);
        u[0]=ActiveCase::PECLET*THERMAL_DIFFUSIVITY/h*sin(PI*Z)*cos(2*PI*s);
    }
    if constexpr (KIND==3)
    {
        const double s=double(VISCOSITY)*step/(h*h);
        u[0]=ActiveCase::CHANNEL_RE*VISCOSITY/h*(1+0.25*sin(2*PI*s))*sin(PI*Z);
    }
#endif
}
__device__ __host__ inline void channelAcceleration(natural_t z,double step,real_t (&a)[3])
{
    a[0]=a[1]=a[2]=0;
#if defined(RBC_VALIDATION_CASE) && RBC_VALIDATION_CASE == 3
    const double h=NZ-1.,s=double(VISCOSITY)*step/(h*h);
    const double f=1+0.25*sin(2*PI*s),fp=0.5*PI*cos(2*PI*s);
    a[0]=VISCOSITY*(ActiveCase::CHANNEL_RE*VISCOSITY/h)/(h*h)*(fp+PI*PI*f)*sin(PI*z/h);
#endif
}
}
