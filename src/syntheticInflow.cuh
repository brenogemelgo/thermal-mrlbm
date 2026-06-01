#pragma once

#include "bitmasks.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

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

__host__ static inline void printSyntheticTurbulentInflowDiagnostics() noexcept
{
    constexpr natural_t radialBins = 8u;
    constexpr real_t activeRadius = ENABLE_SMOOTH_JET_PROFILE
                                        ? RADIUS + JET_INLET_HALO_THICKNESS_CELLS
                                        : RADIUS;

    unsigned long long count = 0;
    double sumX = 0.0;
    double sumY = 0.0;
    double sumZ = 0.0;
    double sumX2 = 0.0;
    double sumY2 = 0.0;
    double sumZ2 = 0.0;
    double maxDu = 0.0;
    double maxVelocity = 0.0;
    double minUz = std::numeric_limits<double>::infinity();
    double maxUz = -std::numeric_limits<double>::infinity();
    double minTemperature = std::numeric_limits<double>::infinity();
    double maxTemperature = -std::numeric_limits<double>::infinity();

    unsigned long long binCount[radialBins] = {};
    double binProfileMin[radialBins];
    double binProfileMax[radialBins];
    double binUzSum[radialBins] = {};
    double binTemperatureSum[radialBins] = {};
    double binDuX2[radialBins] = {};
    double binDuY2[radialBins] = {};
    double binDuZ2[radialBins] = {};

    for (natural_t bin = 0; bin < radialBins; ++bin)
    {
        binProfileMin[bin] = std::numeric_limits<double>::infinity();
        binProfileMax[bin] = -std::numeric_limits<double>::infinity();
    }

    for (natural_t y = 0; y < NY; ++y)
    {
        for (natural_t x = 0; x < NX; ++x)
        {
            const real_t cx0 = static_cast<real_t>(0.5) * static_cast<real_t>(NX - 1u);
            const real_t cy0 = static_cast<real_t>(0.5) * static_cast<real_t>(NY - 1u);
            const real_t dx = static_cast<real_t>(x) - cx0;
            const real_t dy = static_cast<real_t>(y) - cy0;
            const real_t r = turbulentInflowSqrt(dx * dx + dy * dy);

            if (r > activeRadius)
            {
                continue;
            }

            const real_t profile = smoothJetProfile(r);
            const real_t envelope = turbulentInflowEnvelope(profile);
            const real_t temperature = smoothJetTemperature(profile);

            real_t duX;
            real_t duY;
            real_t duZ;
            syntheticTurbulentInflowVelocity(x, y, 0u, envelope, duX, duY, duZ);

            const double ux = static_cast<double>(duX);
            const double uy = static_cast<double>(duY);
            const double uz = std::max(static_cast<double>(smoothJetMeanUz(profile) + duZ), 0.0);
            const double duNorm = std::sqrt(ux * ux + uy * uy + static_cast<double>(duZ) * static_cast<double>(duZ));
            const double uNorm = std::sqrt(ux * ux + uy * uy + uz * uz);

            ++count;
            sumX += ux;
            sumY += uy;
            sumZ += static_cast<double>(duZ);
            sumX2 += ux * ux;
            sumY2 += uy * uy;
            sumZ2 += static_cast<double>(duZ) * static_cast<double>(duZ);
            maxDu = std::max(maxDu, duNorm);
            maxVelocity = std::max(maxVelocity, uNorm);
            minUz = std::min(minUz, uz);
            maxUz = std::max(maxUz, uz);
            minTemperature = std::min(minTemperature, static_cast<double>(temperature));
            maxTemperature = std::max(maxTemperature, static_cast<double>(temperature));

            natural_t bin = static_cast<natural_t>(
                turbulentInflowClamp(r / activeRadius, static_cast<real_t>(0), static_cast<real_t>(0.999999)) *
                static_cast<real_t>(radialBins));
            if (bin >= radialBins)
            {
                bin = radialBins - 1u;
            }

            ++binCount[bin];
            binProfileMin[bin] = std::min(binProfileMin[bin], static_cast<double>(profile));
            binProfileMax[bin] = std::max(binProfileMax[bin], static_cast<double>(profile));
            binUzSum[bin] += uz;
            binTemperatureSum[bin] += static_cast<double>(temperature);
            binDuX2[bin] += ux * ux;
            binDuY2[bin] += uy * uy;
            binDuZ2[bin] += static_cast<double>(duZ) * static_cast<double>(duZ);
        }
    }

    const double invCount = count > 0 ? static_cast<double>(1) / static_cast<double>(count) : 0.0;

    std::cout << "synthetic turbulent inflow diagnostics" << std::endl;
    std::cout << "active inlet nodes: " << count << std::endl;
    std::cout << "modes: " << TURBULENT_INFLOW_MODES << std::endl;
    std::cout << "mean du: "
              << sumX * invCount << ", "
              << sumY * invCount << ", "
              << sumZ * invCount << std::endl;
    std::cout << "rms du: "
              << std::sqrt(sumX2 * invCount) << ", "
              << std::sqrt(sumY2 * invCount) << ", "
              << std::sqrt(sumZ2 * invCount) << std::endl;
    std::cout << "max |du|: " << maxDu << std::endl;
    std::cout << "max |u|: " << maxVelocity << std::endl;
    std::cout << "min/max uz: " << minUz << ", " << maxUz << std::endl;
    std::cout << "min/max temperature: " << minTemperature << ", " << maxTemperature << std::endl;

    std::cout << "profile samples:" << std::endl;
    const real_t sampleRadii[5] = {
        static_cast<real_t>(0),
        static_cast<real_t>(0.5) * RADIUS,
        RADIUS - JET_INLET_SHEAR_THICKNESS_CELLS,
        RADIUS,
        RADIUS + JET_INLET_SHEAR_THICKNESS_CELLS};
    const char *sampleNames[5] = {"0", "0.5R", "R-delta", "R", "R+delta"};
    for (natural_t sample = 0; sample < 5u; ++sample)
    {
        const real_t profile = smoothJetProfile(sampleRadii[sample]);
        std::cout << "  r=" << sampleNames[sample]
                  << ": profile=" << profile
                  << " uzMean=" << smoothJetMeanUz(profile)
                  << " T=" << smoothJetTemperature(profile) << std::endl;
    }

    std::cout << "radial bins:" << std::endl;
    std::cout << "bin,rMin,rMax,count,profileMin,profileMax,meanUz,meanT,rmsDuX,rmsDuY,rmsDuZ" << std::endl;
    for (natural_t bin = 0; bin < radialBins; ++bin)
    {
        const double binInvCount = binCount[bin] > 0 ? 1.0 / static_cast<double>(binCount[bin]) : 0.0;
        const double rMin = static_cast<double>(activeRadius) * static_cast<double>(bin) / static_cast<double>(radialBins);
        const double rMax = static_cast<double>(activeRadius) * static_cast<double>(bin + 1u) / static_cast<double>(radialBins);
        std::cout << bin << ','
                  << rMin << ','
                  << rMax << ','
                  << binCount[bin] << ','
                  << (binCount[bin] > 0 ? binProfileMin[bin] : 0.0) << ','
                  << (binCount[bin] > 0 ? binProfileMax[bin] : 0.0) << ','
                  << binUzSum[bin] * binInvCount << ','
                  << binTemperatureSum[bin] * binInvCount << ','
                  << std::sqrt(binDuX2[bin] * binInvCount) << ','
                  << std::sqrt(binDuY2[bin] * binInvCount) << ','
                  << std::sqrt(binDuZ2[bin] * binInvCount) << std::endl;
    }

    const natural_t cx = NX / 2u;
    const natural_t cy = NY / 2u;
    const natural_t shearX = static_cast<natural_t>(std::min(
        static_cast<double>(NX - 1u),
        0.5 * static_cast<double>(NX - 1u) + static_cast<double>(RADIUS)));
    const natural_t shearY = cy;

    std::cout << "samples at step 0:" << std::endl;
    for (natural_t sample = 0; sample < 2u; ++sample)
    {
        const natural_t sx = sample == 0u ? cx : shearX;
        const natural_t sy = sample == 0u ? cy : shearY;
        const real_t profile = jetInletProfile(sx, sy);
        const real_t envelope = turbulentInflowEnvelope(profile);
        real_t duX;
        real_t duY;
        real_t duZ;
        syntheticTurbulentInflowVelocity(sx, sy, 0u, envelope, duX, duY, duZ);
        const real_t uz = smoothJetMeanUz(profile) + duZ;
        std::cout << "  (" << sx << ", " << sy << "): profile=" << profile
                  << " envelope=" << envelope
                  << " du=(" << duX << ", " << duY << ", " << duZ << ")"
                  << " u=(" << duX << ", " << duY << ", " << (uz > static_cast<real_t>(0) ? uz : static_cast<real_t>(0)) << ")"
                  << " T=" << smoothJetTemperature(profile) << std::endl;
    }

    std::cout << "center temporal samples:" << std::endl;
    const real_t centerProfile = jetInletProfile(cx, cy);
    const real_t centerEnvelope = turbulentInflowEnvelope(centerProfile);
    for (natural_t step = 0; step < TURBULENT_INFLOW_DIAGNOSTIC_STEPS; ++step)
    {
        real_t duX;
        real_t duY;
        real_t duZ;
        syntheticTurbulentInflowVelocity(cx, cy, step, centerEnvelope, duX, duY, duZ);
        std::cout << "  step " << step
                  << ": du=(" << duX << ", " << duY << ", " << duZ << ")"
                  << " uz=" << smoothJetMeanUz(centerProfile) + duZ << std::endl;
    }
}
