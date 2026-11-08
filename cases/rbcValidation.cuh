#pragma once
#include "rayleighBenard.cuh"

// Only the pre-production launcher defines this macro. Production cases are unchanged.
static_assert(RBC_VALIDATION_CASE >= 1 && RBC_VALIDATION_CASE <= 3);
struct RBCValidationCase : RayleighBenardCase
{
    static constexpr real_t RA = 0, PR = 1, NU = 0.02, ALPHA = 0.02, KAPPA = ALPHA;
    static constexpr real_t GBETA = 0;
    static constexpr bool ENABLE_BUOYANCY = false, CONDUCTION_ONLY = true;
    static constexpr real_t PERTURBATION_AMPLITUDE = 0;
    static constexpr real_t U_CHAR = real_t(0.2)*NU/H;
    static constexpr real_t RE = U_CHAR*H/NU;
    // Time, source and velocity scales are diffusive, not RBC free-fall scales.
    static constexpr real_t MODE_AMPLITUDE = 0.1, PECLET = 1;
    static constexpr real_t CHANNEL_RE = 0.2;
    static constexpr natural_t FORMULA_VERSION = 1;
};
