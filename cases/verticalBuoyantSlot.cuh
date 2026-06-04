#pragma once

#include "aux/functions.cuh"

struct VerticalBuoyantSlotCase
{
    static constexpr const char *NAME = "SLOT";

    static constexpr natural_t NX = 96;
    static constexpr natural_t NY = 192;
    static constexpr natural_t NZ = 32;

    static constexpr natural_t NSTEPS = 100000;
    static constexpr natural_t STAMP = 1000;

    static constexpr real_t THOT = static_cast<real_t>(1);
    static constexpr real_t TCOLD = static_cast<real_t>(0);
    static constexpr real_t TREF = static_cast<real_t>(0.5) * (THOT + TCOLD);
    static constexpr real_t T_REF = TREF;
    static constexpr real_t DELTA_T = THOT - TCOLD;

    static constexpr real_t PR = static_cast<real_t>(0.71);
    static constexpr real_t L_SLOT = static_cast<real_t>(NX - 1u);

    static constexpr real_t NU = static_cast<real_t>(0.01);
    static constexpr real_t ALPHA = NU / PR;

    static constexpr real_t RA_TARGET = static_cast<real_t>(2.0e5);

    static constexpr real_t GBETA = RA_TARGET * NU * ALPHA / (DELTA_T * L_SLOT * L_SLOT * L_SLOT);
    static constexpr real_t slotVelocityScale = GBETA * DELTA_T * L_SLOT * L_SLOT / NU;
    static constexpr real_t slotProfileMaxFactor = csqrt<real_t>(static_cast<real_t>(3)) / static_cast<real_t>(216);
    static constexpr real_t analyticalMaxAbsUy = slotVelocityScale * slotProfileMaxFactor;

    static constexpr bool ENABLE_BUOYANCY = true;
    static constexpr bool ENABLE_INLET = false;
    static constexpr bool ENABLE_WALLS = true;
    static constexpr bool ENABLE_OUTFLOW = false;

    static constexpr bool PERIODIC_X = false;
    static constexpr bool PERIODIC_Y = true;
    static constexpr bool PERIODIC_Z = true;

    static constexpr real_t GRAVITY_X = static_cast<real_t>(0);
    static constexpr real_t GRAVITY_Y = static_cast<real_t>(1);
    static constexpr real_t GRAVITY_Z = static_cast<real_t>(0);

    static constexpr real_t RHO0 = static_cast<real_t>(1);
    static constexpr real_t U_CHAR = csqrt<real_t>(GBETA * DELTA_T * L_SLOT);
    static constexpr real_t L_CHAR = L_SLOT;

    static constexpr real_t RE = U_CHAR * L_CHAR / NU;
    static constexpr real_t RA = GBETA * DELTA_T * L_SLOT * L_SLOT * L_SLOT / (NU * ALPHA);
};
