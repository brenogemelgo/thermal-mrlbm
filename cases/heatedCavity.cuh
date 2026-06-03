#pragma once

#include "aux/functions.cuh"

struct HeatedCavityCase
{
    static constexpr const char *NAME = "CAVITY";

    static constexpr natural_t NX = 64;
    static constexpr natural_t NY = 64;
    static constexpr natural_t NZ = 16;

    static constexpr natural_t NSTEPS = 80000; // 30000, 50000, 80000
    static constexpr natural_t STAMP = 1000;

    static constexpr real_t THOT = static_cast<real_t>(1);
    static constexpr real_t TCOLD = static_cast<real_t>(0);
    static constexpr real_t TREF = static_cast<real_t>(0.5) * (THOT + TCOLD);

    static constexpr real_t PR = static_cast<real_t>(0.71);
    static constexpr real_t RA = static_cast<real_t>(1.0e5); // 1e3, 1e4, 1e5

    static constexpr real_t UFREEFALL_TARGET = static_cast<real_t>(0.05);
    static constexpr real_t L_CAVITY = static_cast<real_t>(NX - 1u);
    static constexpr real_t NU = UFREEFALL_TARGET * L_CAVITY * csqrt<real_t>(PR / RA);
    static constexpr real_t ALPHA = NU / PR;

    static constexpr real_t DELTA_T = THOT - TCOLD;
    static constexpr real_t GBETA = RA * NU * ALPHA / (DELTA_T * L_CAVITY * L_CAVITY * L_CAVITY);

    static constexpr bool ENABLE_BUOYANCY = true;
    static constexpr bool ENABLE_INLET = false;
    static constexpr bool ENABLE_WALLS = true;
    static constexpr bool ENABLE_OUTFLOW = false;

    static constexpr bool PERIODIC_X = false;
    static constexpr bool PERIODIC_Y = false;
    static constexpr bool PERIODIC_Z = true;

    static constexpr real_t GRAVITY_X = static_cast<real_t>(0);
    static constexpr real_t GRAVITY_Y = static_cast<real_t>(1);
    static constexpr real_t GRAVITY_Z = static_cast<real_t>(0);

    static constexpr real_t RHO0 = static_cast<real_t>(1);
    static constexpr real_t T_REF = TREF;
    static constexpr real_t U_CHAR = UFREEFALL_TARGET;
    static constexpr real_t L_CHAR = L_CAVITY;
    static constexpr real_t RE = U_CHAR * L_CHAR / NU;
};
