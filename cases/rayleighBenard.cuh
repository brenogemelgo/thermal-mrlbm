#pragma once
#include "aux/functions.cuh"

// Compile-time production selection and independent overrides for short tests.
#ifndef RBC_CASE
#define RBC_CASE 1
#endif
#ifndef RBC_N
#define RBC_N 64
#endif
#ifndef RBC_RA
#define RBC_RA (RBC_CASE == 1 ? 1.e6 : RBC_CASE == 2 ? 1.e7 : 1.e8)
#endif
#ifndef RBC_PR
#define RBC_PR (RBC_CASE == 4 ? 1.0 : 0.7)
#endif
#ifndef RBC_SAMPLE
#define RBC_SAMPLE 100
#endif
#ifndef RBC_STAMP
#define RBC_STAMP 5000
#endif
struct RayleighBenardCase
{
    static_assert(RBC_CASE >= 1 && RBC_CASE <= 4);
    static constexpr const char *NAME = "RAYLEIGH_BENARD";
    static constexpr natural_t CASE_ID = RBC_CASE;
    static constexpr natural_t NX = RBC_N, NY = RBC_N, NZ = RBC_N;
    static_assert(NZ >= 4);
    static constexpr real_t RA = RBC_RA, PR = RBC_PR;
    static_assert(RA >= 0 && PR > 0);
    static constexpr bool CONDUCTION_ONLY = RA == 0;
    static constexpr real_t THOT = 1, TCOLD = 0, DELTA_T = THOT - TCOLD;
    static constexpr real_t T0 = 0.5, T_REF = T0, RHO0 = 1;
    // Plates are nodes z=0 and z=NZ-1. Periodic lengths are NX and NY.
    static constexpr real_t H = NZ - 1u;
    static constexpr real_t ASPECT_RATIO = static_cast<real_t>(NX) / H;
    static constexpr real_t U_FREEFALL = 0.05;
    static constexpr double T_FREEFALL = static_cast<double>(H) / U_FREEFALL;
    static constexpr real_t NU = CONDUCTION_ONLY ? real_t(0.01) :
        U_FREEFALL * H * csqrt<real_t>(PR / (RA > 0 ? RA : real_t(1)));
    static constexpr real_t ALPHA = NU / PR, KAPPA = ALPHA;
    static constexpr real_t GBETA = CONDUCTION_ONLY ? 0 :
        U_FREEFALL * U_FREEFALL / (DELTA_T * H);
    static constexpr double TRANSIENT_FF = RBC_CASE <= 2 ? 100 : 200;
    static constexpr double AVERAGING_FF = RBC_CASE == 1 ? 150 : RBC_CASE == 2 ? 200 : 300;
#ifdef RBC_TRANSIENT_STEPS
    static constexpr natural_t timeAverageStart = RBC_TRANSIENT_STEPS;
#else
    static constexpr natural_t timeAverageStart = static_cast<natural_t>(TRANSIENT_FF * T_FREEFALL + 0.5);
#endif
#ifdef RBC_STEPS
    static constexpr natural_t NSTEPS = RBC_STEPS;
#else
    static constexpr natural_t NSTEPS = timeAverageStart + static_cast<natural_t>(AVERAGING_FF * T_FREEFALL + 0.5);
#endif
    static constexpr natural_t STAMP = RBC_STAMP, timeAverageStride = RBC_SAMPLE;
    static_assert(STAMP > 0 && timeAverageStride > 0);
    static constexpr bool timeAverageEnabled = true;
    static constexpr real_t PI = 3.14159265358979323846, TWO_PI = 2 * PI;
#ifndef RBC_PERTURBATION
#define RBC_PERTURBATION 1.e-3
#endif
    static constexpr real_t EPSILON_T = RBC_PERTURBATION;
    static constexpr real_t PERTURBATION_AMPLITUDE = CONDUCTION_ONLY ? 0 : EPSILON_T;
    static constexpr bool ENABLE_BUOYANCY = !CONDUCTION_ONLY;
    static constexpr bool ENABLE_INLET = false, ENABLE_WALLS = true, ENABLE_OUTFLOW = false;
    static constexpr bool PERIODIC_X = true, PERIODIC_Y = true, PERIODIC_Z = false;
    static constexpr const char *verticalDirection = "z";
    static constexpr real_t GRAVITY_X = 0, GRAVITY_Y = 0, GRAVITY_Z = 1;
    static constexpr real_t U_CHAR = U_FREEFALL, L_CHAR = H, RE = U_CHAR * H / NU;
};
