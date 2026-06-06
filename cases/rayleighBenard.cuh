#pragma once

#include "aux/functions.cuh"

struct RayleighBenardCase
{
    static constexpr const char *NAME = "RAYLEIGH_BENARD";

    static constexpr natural_t NX = 256;
    static constexpr natural_t NY = 16;
    static constexpr natural_t NZ = 128;

    static constexpr natural_t NSTEPS = 100000;
    static constexpr natural_t STAMP = 1000;

    static constexpr real_t RA = static_cast<real_t>(0);
    static constexpr real_t PR = static_cast<real_t>(0.71);

    static constexpr real_t THOT = static_cast<real_t>(1);
    static constexpr real_t TCOLD = static_cast<real_t>(0);
    static constexpr real_t T0 = static_cast<real_t>(0.5) * (THOT + TCOLD);
    static constexpr real_t T_REF = T0;
    static constexpr real_t DELTA_T = THOT - TCOLD;

    static constexpr real_t H = static_cast<real_t>(NZ - 1u);
    static constexpr real_t ASPECT_RATIO = static_cast<real_t>(NX) / H;

    // Set RA=0 for pure conduction, or RA=1e4, 1e5, 1e6 for convection.
    static constexpr bool CONDUCTION_ONLY = RA == static_cast<real_t>(0);
    static constexpr real_t NU = static_cast<real_t>(0.01);
    static constexpr real_t ALPHA = NU / PR;
    static constexpr real_t GBETA = CONDUCTION_ONLY
                                        ? static_cast<real_t>(0)
                                        : RA * NU * ALPHA / (DELTA_T * H * H * H);

    static constexpr real_t PI = static_cast<real_t>(3.14159265358979323846);
    static constexpr real_t TWO_PI = static_cast<real_t>(6.28318530717958647692);
    static constexpr real_t EPSILON_T = static_cast<real_t>(1.0e-3);
    static constexpr real_t PERTURBATION_AMPLITUDE = CONDUCTION_ONLY ? static_cast<real_t>(0) : EPSILON_T;

    static constexpr bool ENABLE_BUOYANCY = true;
    static constexpr bool ENABLE_INLET = false;
    static constexpr bool ENABLE_WALLS = true;
    static constexpr bool ENABLE_OUTFLOW = false;

    static constexpr bool PERIODIC_X = true;
    static constexpr bool PERIODIC_Y = true;
    static constexpr bool PERIODIC_Z = false;

    static constexpr real_t GRAVITY_X = static_cast<real_t>(0);
    static constexpr real_t GRAVITY_Y = static_cast<real_t>(0);
    static constexpr real_t GRAVITY_Z = static_cast<real_t>(1);

    static constexpr real_t RHO0 = static_cast<real_t>(1);
    static constexpr real_t U_FREEFALL = CONDUCTION_ONLY
                                             ? static_cast<real_t>(0)
                                             : csqrt<real_t>(GBETA * DELTA_T * H);
    static constexpr real_t U_CHAR = U_FREEFALL;
    static constexpr real_t L_CHAR = H;
    static constexpr real_t RE = U_CHAR * L_CHAR / NU;
};
