#pragma once

// =================================================================================================== //

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <cuda_fp16.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <type_traits>
#include <vector>
#include <stdexcept>
#include <string>

#define D3Q27

// =================================================================================================== //

using natural_t = uint32_t;
using real_t = float;

// =================================================================================================== //
// USER PARAMETERS

// --------------------------------------------------------------------------------------------------- //
// Hydro/reference scales

#ifndef L_CHAR_VALUE
#define L_CHAR_VALUE 256
#endif

#ifndef U_CHAR_VALUE
#define U_CHAR_VALUE 0.05
#endif

#ifndef REYNOLDS_VALUE
#define REYNOLDS_VALUE 10000
#endif

#ifndef NSTEPS_VALUE
#define NSTEPS_VALUE 500000
#endif

#ifndef STAMP_VALUE
#define STAMP_VALUE 10000
#endif

// --------------------------------------------------------------------------------------------------- //
// Hydrodynamic two-relaxation-time split
//
// BULK_RELAXATION_FACTOR_VALUE = 1.0 gives tau_bulk = tau_nu, i.e. BGK-equivalent trace relaxation.
// BULK_RELAXATION_FACTOR_VALUE > 1.0 relaxes the bulk/trace mode more slowly.
// BULK_RELAXATION_FACTOR_VALUE < 1.0 relaxes the bulk/trace mode faster, but must keep tau_bulk > 0.5.

#ifndef BULK_RELAXATION_FACTOR_VALUE
#define BULK_RELAXATION_FACTOR_VALUE 1.0
#endif

// --------------------------------------------------------------------------------------------------- //
// Thermal physical/numerical parameters

#ifndef PRANDTL_VALUE
#define PRANDTL_VALUE 0.71
#endif

#ifndef T_COLD_VALUE
#define T_COLD_VALUE 0.0
#endif

#ifndef T_HOT_VALUE
#define T_HOT_VALUE 1.0
#endif

#ifndef T_REF_VALUE
#define T_REF_VALUE 0.5
#endif

#ifndef BETA_THERMAL_VALUE
#define BETA_THERMAL_VALUE 1.0
#endif

#ifndef THERMAL_SOURCE_VALUE_PARAM
#define THERMAL_SOURCE_VALUE_PARAM 0.0
#endif

#ifndef THERMAL_GRAVITY_X_VALUE
#define THERMAL_GRAVITY_X_VALUE 0.0
#endif

#ifndef THERMAL_GRAVITY_Y_VALUE
#define THERMAL_GRAVITY_Y_VALUE 0.0
#endif

#ifndef THERMAL_GRAVITY_Z_VALUE
#define THERMAL_GRAVITY_Z_VALUE -1.0
#endif

#ifndef THERMAL_RHO0_VALUE
#define THERMAL_RHO0_VALUE 1.0
#endif

// D3Q7 thermal lattice convention:
// w0 = 1/4, waxis = 1/8, therefore c_sT^2 = 1/4.
#ifndef THERMAL_CS2_VALUE
#define THERMAL_CS2_VALUE 0.25
#endif

// --------------------------------------------------------------------------------------------------- //
// CUDA launch geometry

#ifndef BLOCK_NX_VALUE
#define BLOCK_NX_VALUE 32
#endif

#ifndef BLOCK_NY_VALUE
#define BLOCK_NY_VALUE 4
#endif

#ifndef BLOCK_NZ_VALUE
#define BLOCK_NZ_VALUE 4
#endif

// =================================================================================================== //
// Typed constants

constexpr natural_t L_CHAR = static_cast<natural_t>(L_CHAR_VALUE);
constexpr real_t U_CHAR = static_cast<real_t>(U_CHAR_VALUE);
constexpr real_t REYNOLDS = static_cast<real_t>(REYNOLDS_VALUE);
constexpr natural_t NSTEPS = static_cast<natural_t>(NSTEPS_VALUE);
constexpr natural_t STAMP = static_cast<natural_t>(STAMP_VALUE);

// =================================================================================================== //
// Hydrodynamic relaxation

constexpr real_t VISCOSITY = static_cast<real_t>((static_cast<double>(U_CHAR) * static_cast<double>(L_CHAR)) / static_cast<double>(REYNOLDS));
constexpr real_t TAU_NU = static_cast<real_t>(0.5) + static_cast<real_t>(3.0) * VISCOSITY;
constexpr real_t TAU = TAU_NU;
constexpr real_t OMEGA = static_cast<real_t>(static_cast<double>(1) / static_cast<double>(TAU));
constexpr real_t T_OMEGA = static_cast<real_t>(1.0) - OMEGA;
constexpr real_t OMEGA_D2 = static_cast<real_t>(0.5) * OMEGA;

constexpr real_t BULK_RELAXATION_FACTOR = static_cast<real_t>(BULK_RELAXATION_FACTOR_VALUE);
constexpr real_t TAU_BULK = static_cast<real_t>(0.5) + BULK_RELAXATION_FACTOR * (TAU_NU - static_cast<real_t>(0.5));
constexpr real_t OMEGA_BULK = static_cast<real_t>(static_cast<double>(1) / static_cast<double>(TAU_BULK));
constexpr real_t T_OMEGA_BULK = static_cast<real_t>(1.0) - OMEGA_BULK;

// =================================================================================================== //
// Thermal parameters

constexpr real_t PRANDTL = static_cast<real_t>(PRANDTL_VALUE);
constexpr real_t T_COLD = static_cast<real_t>(T_COLD_VALUE);
constexpr real_t T_HOT = static_cast<real_t>(T_HOT_VALUE);
constexpr real_t T_REF = static_cast<real_t>(T_REF_VALUE);
constexpr real_t BETA_THERMAL = static_cast<real_t>(BETA_THERMAL_VALUE);
constexpr real_t THERMAL_SOURCE_VALUE = static_cast<real_t>(THERMAL_SOURCE_VALUE_PARAM);
constexpr real_t THERMAL_GRAVITY_X = static_cast<real_t>(THERMAL_GRAVITY_X_VALUE);
constexpr real_t THERMAL_GRAVITY_Y = static_cast<real_t>(THERMAL_GRAVITY_Y_VALUE);
constexpr real_t THERMAL_GRAVITY_Z = static_cast<real_t>(THERMAL_GRAVITY_Z_VALUE);
constexpr real_t THERMAL_RHO0 = static_cast<real_t>(THERMAL_RHO0_VALUE);
constexpr real_t THERMAL_CS2 = static_cast<real_t>(THERMAL_CS2_VALUE);

constexpr real_t THERMAL_FORCE_SCALE =
    static_cast<real_t>((static_cast<double>(U_CHAR) * static_cast<double>(U_CHAR)) /
                        static_cast<double>(L_CHAR));

constexpr real_t THERMAL_DIFFUSIVITY = VISCOSITY / PRANDTL;
constexpr real_t TAU_T = static_cast<real_t>(0.5) + THERMAL_DIFFUSIVITY / THERMAL_CS2;
constexpr real_t OMEGA_T = static_cast<real_t>(static_cast<double>(1) / static_cast<double>(TAU_T));
constexpr real_t T_OMEGA_T = static_cast<real_t>(1.0) - OMEGA_T;

// =================================================================================================== //
// Domain

constexpr natural_t NX = L_CHAR;
constexpr natural_t NY = L_CHAR;
constexpr natural_t NZ = L_CHAR;

constexpr natural_t CELLS = NX * NY * NZ;
constexpr natural_t STRIDE = NX * NY;

// =================================================================================================== //
// CUDA launch geometry

constexpr natural_t BLOCK_NX = static_cast<natural_t>(BLOCK_NX_VALUE);
constexpr natural_t BLOCK_NY = static_cast<natural_t>(BLOCK_NY_VALUE);
constexpr natural_t BLOCK_NZ = static_cast<natural_t>(BLOCK_NZ_VALUE);

constexpr natural_t GRID_X = (NX + BLOCK_NX - 1) / BLOCK_NX;
constexpr natural_t GRID_Y = (NY + BLOCK_NY - 1) / BLOCK_NY;
constexpr natural_t GRID_Z = (NZ + BLOCK_NZ - 1) / BLOCK_NZ;

// =================================================================================================== //
// Moment layout

constexpr natural_t HYDRO_MOMENTS = 10;

// Thermal MR storage:
// TEMP, TFX, TFY, TFZ.
// D3Q7 populations are reconstructed locally, not stored.
constexpr natural_t THERMAL_MOMENTS = 4;

constexpr natural_t NUM_MOMENTS = HYDRO_MOMENTS + THERMAL_MOMENTS;

// =================================================================================================== //
// Moment indices

constexpr natural_t RHO = 0;
constexpr natural_t UX = 1;
constexpr natural_t UY = 2;
constexpr natural_t UZ = 3;

constexpr natural_t MXX = 4;
constexpr natural_t MYY = 5;
constexpr natural_t MZZ = 6;
constexpr natural_t MXY = 7;
constexpr natural_t MXZ = 8;
constexpr natural_t MYZ = 9;

constexpr natural_t TEMP = 10;
constexpr natural_t TFX = 11;
constexpr natural_t TFY = 12;
constexpr natural_t TFZ = 13;

// =================================================================================================== //
// Compile-time sanity checks

static_assert(L_CHAR > 0, "L_CHAR must be positive.");
static_assert(U_CHAR > static_cast<real_t>(0), "U_CHAR must be positive.");
static_assert(REYNOLDS > static_cast<real_t>(0), "REYNOLDS must be positive.");
static_assert(PRANDTL > static_cast<real_t>(0), "PRANDTL must be positive.");

static_assert(TAU_NU > static_cast<real_t>(0.5), "TAU_NU must be larger than 0.5.");
static_assert(TAU_BULK > static_cast<real_t>(0.5), "TAU_BULK must be larger than 0.5.");
static_assert(TAU_T > static_cast<real_t>(0.5), "TAU_T must be larger than 0.5.");

static_assert(BLOCK_NX > 0, "BLOCK_NX must be positive.");
static_assert(BLOCK_NY > 0, "BLOCK_NY must be positive.");
static_assert(BLOCK_NZ > 0, "BLOCK_NZ must be positive.");

static_assert(HYDRO_MOMENTS == 10, "Hydro MR layout expects 10 moments.");
static_assert(NUM_MOMENTS == 14, "Thermal MR layout expects 14 moments.");

// =================================================================================================== //
