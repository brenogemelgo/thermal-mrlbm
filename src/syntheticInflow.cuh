#pragma once

#include "bitmasks.cuh"

#include <cmath>

constexpr real_t TURBULENT_INFLOW_TWO_PI = static_cast<real_t>(6.2831853071795864769);

__host__ __device__ [[nodiscard]] static inline uint32_t turbulentInflowHash32(
    uint32_t value) noexcept
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

__host__ __device__ [[nodiscard]] static inline real_t turbulentInflowRandom01(
    const natural_t mode,
    const natural_t selector) noexcept
{
    const uint32_t mixed =
        static_cast<uint32_t>(TURBULENT_INFLOW_SEED) ^
        (static_cast<uint32_t>(mode) * 0x9e3779b9u) ^
        (static_cast<uint32_t>(selector) * 0x85ebca6bu);

    return static_cast<real_t>(turbulentInflowHash32(mixed) >> 8) *
           static_cast<real_t>(1.0 / 16777216.0);
}

__host__ __device__ [[nodiscard]] static inline real_t turbulentInflowSin(
    const real_t value) noexcept
{
#ifdef __CUDA_ARCH__
    return __sinf(value);
#else
    return static_cast<real_t>(std::sin(static_cast<double>(value)));
#endif
}

__host__ __device__ [[nodiscard]] static inline real_t turbulentInflowCos(
    const real_t value) noexcept
{
#ifdef __CUDA_ARCH__
    return __cosf(value);
#else
    return static_cast<real_t>(std::cos(static_cast<double>(value)));
#endif
}

__host__ __device__ [[nodiscard]] static inline real_t turbulentInflowTanh(
    const real_t value) noexcept
{
#ifdef __CUDA_ARCH__
    return tanhf(value);
#else
    return static_cast<real_t>(std::tanh(static_cast<double>(value)));
#endif
}

__host__ __device__ [[nodiscard]] static inline real_t turbulentInflowSqrt(
    const real_t value) noexcept
{
#ifdef __CUDA_ARCH__
    return sqrtf(value);
#else
    return static_cast<real_t>(std::sqrt(static_cast<double>(value)));
#endif
}

__host__ __device__ [[nodiscard]] static inline real_t turbulentInflowLog(
    const real_t value) noexcept
{
#ifdef __CUDA_ARCH__
    return logf(value);
#else
    return static_cast<real_t>(std::log(static_cast<double>(value)));
#endif
}

__host__ __device__ [[nodiscard]] static inline real_t turbulentInflowExp(
    const real_t value) noexcept
{
#ifdef __CUDA_ARCH__
    return expf(value);
#else
    return static_cast<real_t>(std::exp(static_cast<double>(value)));
#endif
}

__host__ __device__ [[nodiscard]] static inline real_t turbulentInflowClamp(
    const real_t value,
    const real_t lo,
    const real_t hi) noexcept
{
    return value < lo ? lo : (value > hi ? hi : value);
}

__host__ __device__ [[nodiscard]] static inline real_t smoothJetProfile(
    const real_t r) noexcept
{
    if constexpr (!ENABLE_SMOOTH_JET_PROFILE)
    {
        return r <= RADIUS ? static_cast<real_t>(1) : static_cast<real_t>(0);
    }
    else
    {
        const real_t delta = JET_INLET_SHEAR_THICKNESS_CELLS > static_cast<real_t>(1.0e-6)
                                 ? JET_INLET_SHEAR_THICKNESS_CELLS
                                 : static_cast<real_t>(1.0e-6);
        const real_t profile = static_cast<real_t>(0.5) *
                               (static_cast<real_t>(1) - turbulentInflowTanh((r - RADIUS) / delta));
        return turbulentInflowClamp(profile, static_cast<real_t>(0), static_cast<real_t>(1));
    }
}

__host__ __device__ [[nodiscard]] static inline real_t jetInletProfile(
    const natural_t x,
    const natural_t y) noexcept
{
    const real_t cx0 = static_cast<real_t>(0.5) * static_cast<real_t>(NX - 1u);
    const real_t cy0 = static_cast<real_t>(0.5) * static_cast<real_t>(NY - 1u);
    const real_t dx = static_cast<real_t>(x) - cx0;
    const real_t dy = static_cast<real_t>(y) - cy0;
    const real_t r = turbulentInflowSqrt(dx * dx + dy * dy);

    return smoothJetProfile(r);
}

__host__ __device__ [[nodiscard]] static inline real_t smoothJetMeanUz(
    const real_t profile) noexcept
{
    return JET_AMBIENT_UZ + profile * (U_CHAR - JET_AMBIENT_UZ);
}

__host__ __device__ [[nodiscard]] static inline real_t smoothJetTemperature(
    const real_t profile) noexcept
{
    if constexpr (SMOOTH_INLET_TEMPERATURE)
    {
        return T_COLD + profile * (T_HOT - T_COLD);
    }
    else
    {
        return T_HOT;
    }
}

__host__ __device__ [[nodiscard]] static inline real_t turbulentInflowEnvelope(
    const real_t profile) noexcept
{
    const real_t p = turbulentInflowClamp(profile, static_cast<real_t>(0), static_cast<real_t>(1));

    if constexpr (TURBULENT_INFLOW_ENVELOPE_MODE == 1u)
    {
        return turbulentInflowSqrt(p);
    }
    else if constexpr (TURBULENT_INFLOW_ENVELOPE_MODE == 2u)
    {
        return turbulentInflowClamp(static_cast<real_t>(4) * p * (static_cast<real_t>(1) - p),
                                    static_cast<real_t>(0),
                                    static_cast<real_t>(1));
    }
    else
    {
        return p;
    }
}

__host__ __device__ static inline void turbulentInflowModeContribution(
    const natural_t mode,
    const real_t x,
    const real_t y,
    const real_t zVirtual,
    real_t &duX,
    real_t &duY,
    real_t &duZ) noexcept
{
    const real_t kzRaw = static_cast<real_t>(2) * turbulentInflowRandom01(mode, 0u) - static_cast<real_t>(1);
    const real_t theta = TURBULENT_INFLOW_TWO_PI * turbulentInflowRandom01(mode, 1u);
    const real_t radial = turbulentInflowSqrt(static_cast<real_t>(1) - kzRaw * kzRaw);

    const real_t khx = radial * turbulentInflowCos(theta);
    const real_t khy = radial * turbulentInflowSin(theta);
    const real_t khz = kzRaw;

    real_t ax = static_cast<real_t>(2) * turbulentInflowRandom01(mode, 2u) - static_cast<real_t>(1);
    real_t ay = static_cast<real_t>(2) * turbulentInflowRandom01(mode, 3u) - static_cast<real_t>(1);
    real_t az = static_cast<real_t>(2) * turbulentInflowRandom01(mode, 4u) - static_cast<real_t>(1);

    real_t ex = khy * az - khz * ay;
    real_t ey = khz * ax - khx * az;
    real_t ez = khx * ay - khy * ax;
    real_t eNorm2 = ex * ex + ey * ey + ez * ez;

    if (eNorm2 < static_cast<real_t>(1.0e-8))
    {
        ax = static_cast<real_t>(0);
        ay = static_cast<real_t>(1);
        az = static_cast<real_t>(0);
        ex = khy * az - khz * ay;
        ey = khz * ax - khx * az;
        ez = khx * ay - khy * ax;
        eNorm2 = ex * ex + ey * ey + ez * ez;
    }

    const real_t invENorm = static_cast<real_t>(1) / turbulentInflowSqrt(eNorm2);
    ex *= invENorm;
    ey *= invENorm;
    ez *= invENorm;

    constexpr real_t jetDiameter = static_cast<real_t>(2) * RADIUS;
    constexpr real_t lambdaMin = TURBULENT_INFLOW_LMIN_CELLS;
    constexpr real_t lambdaMax = TURBULENT_INFLOW_LMAX_DIAM * jetDiameter;
    const real_t modeFraction =
        (static_cast<real_t>(mode) + static_cast<real_t>(0.5)) /
        static_cast<real_t>(TURBULENT_INFLOW_MODES);
    const real_t lambda = turbulentInflowExp(
        turbulentInflowLog(lambdaMax) +
        modeFraction * (turbulentInflowLog(lambdaMin) - turbulentInflowLog(lambdaMax)));
    const real_t waveNumber = TURBULENT_INFLOW_TWO_PI / lambda;

    const real_t phase = waveNumber * (khx * x + khy * y + khz * zVirtual) +
                         TURBULENT_INFLOW_TWO_PI * turbulentInflowRandom01(mode, 5u);
    const real_t amplitude = turbulentInflowSqrt(
        static_cast<real_t>(2) / static_cast<real_t>(TURBULENT_INFLOW_MODES));
    const real_t c = amplitude * turbulentInflowCos(phase);

    duX += ex * c;
    duY += ey * c;
    duZ += ez * c;
}

__host__ __device__ static inline void syntheticTurbulentInflowVelocity(
    const natural_t x,
    const natural_t y,
    const natural_t step,
    const real_t radialEnvelope,
    real_t &duX,
    real_t &duY,
    real_t &duZ) noexcept
{
    duX = static_cast<real_t>(0);
    duY = static_cast<real_t>(0);
    duZ = static_cast<real_t>(0);

    if constexpr (!ENABLE_SYNTHETIC_TURBULENT_INFLOW)
    {
        return;
    }

    if (radialEnvelope <= static_cast<real_t>(0))
    {
        return;
    }

    const real_t cx0 = static_cast<real_t>(0.5) * static_cast<real_t>(NX - 1u);
    const real_t cy0 = static_cast<real_t>(0.5) * static_cast<real_t>(NY - 1u);
    const real_t dx = static_cast<real_t>(x) - cx0;
    const real_t dy = static_cast<real_t>(y) - cy0;
    const real_t zVirtual = -TURBULENT_INFLOW_CONVECTION_SPEED * static_cast<real_t>(step);

#pragma unroll
    for (natural_t mode = 0; mode < TURBULENT_INFLOW_MODES; ++mode)
    {
        turbulentInflowModeContribution(mode, dx, dy, zVirtual, duX, duY, duZ);
    }

    const real_t r = turbulentInflowSqrt(dx * dx + dy * dy);
    const real_t invR = r > static_cast<real_t>(1.0e-6) ? static_cast<real_t>(1) / r : static_cast<real_t>(0);
    const real_t erX = dx * invR;
    const real_t erY = dy * invR;
    const real_t etX = -erY;
    const real_t etY = erX;

    real_t duR = duX * erX + duY * erY;
    real_t duT = duX * etX + duY * etY;
    real_t duA = duZ;

    duR *= TURBULENT_INFLOW_INTENSITY_RADIAL * U_CHAR * radialEnvelope;
    duT *= TURBULENT_INFLOW_INTENSITY_AZIMUTHAL * U_CHAR * radialEnvelope;
    duA *= TURBULENT_INFLOW_INTENSITY_AXIAL * U_CHAR * radialEnvelope;

    duX = duR * erX + duT * etX;
    duY = duR * erY + duT * etY;
    duZ = duA;

    const real_t mag2 = duX * duX + duY * duY + duZ * duZ;
    constexpr real_t maxMag = TURBULENT_INFLOW_MAX_PERTURBATION;
    if (mag2 > maxMag * maxMag)
    {
        const real_t scale = maxMag / turbulentInflowSqrt(mag2);
        duX *= scale;
        duY *= scale;
        duZ *= scale;
    }

    const real_t minUzPerturbation = -static_cast<real_t>(0.95) * U_CHAR;
    if (duZ < minUzPerturbation)
    {
        duZ = minUzPerturbation;
    }
}
