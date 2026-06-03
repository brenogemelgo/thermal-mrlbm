#pragma once

struct FourierAdvectionDiffusionCase
{
    static constexpr const char *NAME = "FOURIER";

    static constexpr natural_t NX = 32;
    static constexpr natural_t NY = 32;
    static constexpr natural_t NZ = 32;

    static constexpr natural_t NSTEPS = 5000;
    static constexpr natural_t STAMP = 250;

    static constexpr real_t T0 = static_cast<real_t>(1);
    static constexpr real_t AMP_T = static_cast<real_t>(1.0e-2);
    static constexpr real_t THOT = T0 + AMP_T;
    static constexpr real_t TCOLD = T0 - AMP_T;

    static constexpr real_t UX0 = static_cast<real_t>(0.02);
    static constexpr real_t UY0 = static_cast<real_t>(0.01);
    static constexpr real_t UZ0 = static_cast<real_t>(0.015);

    static constexpr natural_t MODE_X = 1;
    static constexpr natural_t MODE_Y = 1;
    static constexpr natural_t MODE_Z = 1;

    static constexpr real_t TWO_PI = static_cast<real_t>(6.2831853071795864769);
    static constexpr real_t KX = TWO_PI * static_cast<real_t>(MODE_X) / static_cast<real_t>(NX);
    static constexpr real_t KY = TWO_PI * static_cast<real_t>(MODE_Y) / static_cast<real_t>(NY);
    static constexpr real_t KZ = TWO_PI * static_cast<real_t>(MODE_Z) / static_cast<real_t>(NZ);
    static constexpr real_t K2 = KX * KX + KY * KY + KZ * KZ;

    static constexpr real_t NU = static_cast<real_t>(0.01);
    static constexpr real_t ALPHA = static_cast<real_t>(0.005);
    static constexpr real_t PR = NU / ALPHA;
    static constexpr real_t DECAY_RATE = ALPHA * K2;

    static constexpr bool ENABLE_BUOYANCY = false;
    static constexpr bool ENABLE_INLET = false;
    static constexpr bool ENABLE_WALLS = false;
    static constexpr bool ENABLE_OUTFLOW = false;

    static constexpr bool PERIODIC_X = true;
    static constexpr bool PERIODIC_Y = true;
    static constexpr bool PERIODIC_Z = true;

    static constexpr real_t GRAVITY_X = static_cast<real_t>(0);
    static constexpr real_t GRAVITY_Y = static_cast<real_t>(0);
    static constexpr real_t GRAVITY_Z = static_cast<real_t>(0);

    static constexpr real_t RHO0 = static_cast<real_t>(1);
    static constexpr real_t T_REF = T0;
    static constexpr real_t GBETA = static_cast<real_t>(0);

    static constexpr real_t U_CHAR = UX0;
    static constexpr real_t L_CHAR = static_cast<real_t>(NX);
    static constexpr real_t RE = U_CHAR * L_CHAR / NU;
};
