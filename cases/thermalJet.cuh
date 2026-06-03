#pragma once

struct ThermalJetCase
{
    static constexpr const char *NAME = "JET";

    static constexpr natural_t NX = 128;
    static constexpr natural_t NY = 128;
    static constexpr natural_t NZ = 256;

    static constexpr natural_t NSTEPS = 100000;
    static constexpr natural_t STAMP = 1000;

    static constexpr real_t U_JET = static_cast<real_t>(0.01);
    static constexpr real_t D_JET = static_cast<real_t>(20);

    static constexpr real_t RE = static_cast<real_t>(5000);
    static constexpr real_t PR = static_cast<real_t>(0.71);
    static constexpr real_t RI = static_cast<real_t>(1);

    static constexpr real_t T_INLET = static_cast<real_t>(1);
    static constexpr real_t T_AMBIENT = static_cast<real_t>(0);
    static constexpr real_t THOT = T_INLET;
    static constexpr real_t TCOLD = T_AMBIENT;
    static constexpr real_t T_REF = T_AMBIENT;
    static constexpr real_t DELTA_T = T_INLET - T_AMBIENT;

    static constexpr real_t NU = U_JET * D_JET / RE;
    static constexpr real_t ALPHA = NU / PR;
    static constexpr real_t GBETA = RI * U_JET * U_JET / (DELTA_T * D_JET);

    static constexpr bool ENABLE_BUOYANCY = RI != static_cast<real_t>(0);
    static constexpr bool ENABLE_INLET = true;
    static constexpr bool ENABLE_WALLS = true;
    static constexpr bool ENABLE_OUTFLOW = true;

    static constexpr bool PERIODIC_X = true;
    static constexpr bool PERIODIC_Y = true;
    static constexpr bool PERIODIC_Z = false;

    static constexpr real_t GRAVITY_X = static_cast<real_t>(0);
    static constexpr real_t GRAVITY_Y = static_cast<real_t>(0);
    static constexpr real_t GRAVITY_Z = static_cast<real_t>(1);

    static constexpr real_t RHO0 = static_cast<real_t>(1);
    static constexpr real_t U_CHAR = U_JET;
    static constexpr real_t L_CHAR = D_JET;
};
