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

#define L_CHAR_VALUE 256
#define U_CHAR_VALUE 0.05
#define REYNOLDS_VALUE 10000
#define NSTEPS_VALUE 500000
#define STAMP_VALUE 10000

// --------------------------------------------------------------------------------------------------- //
// Hydrodynamic two-relaxation-time split
//
// BULK_RELAXATION_FACTOR_VALUE = 1.0 gives tau_bulk = tau_nu, i.e. BGK-equivalent trace relaxation.
// BULK_RELAXATION_FACTOR_VALUE > 1.0 relaxes the bulk/trace mode more slowly.
// BULK_RELAXATION_FACTOR_VALUE < 1.0 relaxes the bulk/trace mode faster, but must keep tau_bulk > 0.5.

#define BULK_RELAXATION_FACTOR_VALUE 1.0

// --------------------------------------------------------------------------------------------------- //
// Thermal model switches
//
// ENABLE_THERMAL_VALUE:
//   0 = hydro only
//   1 = hydro + D3Q7 thermal MR extension
//
// THERMAL_FORCING_ENABLED_VALUE:
//   0 = passive temperature field, no thermal feedback on hydro
//   1 = enable Boussinesq force
//
// THERMAL_SOURCE_ENABLED_VALUE:
//   0 = no volumetric thermal source
//   1 = enable uniform source THERMAL_SOURCE_VALUE_PARAM

#define ENABLE_THERMAL_VALUE 0
#define THERMAL_FORCING_ENABLED_VALUE 0
#define THERMAL_SOURCE_ENABLED_VALUE 0

// --------------------------------------------------------------------------------------------------- //
// Thermal physical/numerical parameters

#define PRANDTL_VALUE 0.71

#define T_COLD_VALUE 0.0
#define T_HOT_VALUE 1.0
#define T_REF_VALUE 0.5

#define BETA_THERMAL_VALUE 0.0

#define THERMAL_SOURCE_VALUE_PARAM 0.0

#define THERMAL_GRAVITY_X_VALUE 0.0
#define THERMAL_GRAVITY_Y_VALUE 0.0
#define THERMAL_GRAVITY_Z_VALUE 0.0

#define THERMAL_RHO0_VALUE 1.0

// D3Q7 thermal lattice convention:
// w0 = 1/4, waxis = 1/8, therefore c_sT^2 = 1/4.
#define THERMAL_CS2_VALUE 0.25

// --------------------------------------------------------------------------------------------------- //
// CUDA launch geometry

#define BLOCK_NX_VALUE 32
#define BLOCK_NY_VALUE 4
#define BLOCK_NZ_VALUE 4

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

constexpr bool ENABLE_THERMAL = static_cast<bool>(ENABLE_THERMAL_VALUE);
constexpr bool THERMAL_FORCING_ENABLED = static_cast<bool>(THERMAL_FORCING_ENABLED_VALUE);
constexpr bool THERMAL_SOURCE_ENABLED = static_cast<bool>(THERMAL_SOURCE_ENABLED_VALUE);
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
constexpr natural_t THERMAL_MOMENTS = ENABLE_THERMAL ? 4 : 0;

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

static_assert(ENABLE_THERMAL_VALUE == 0 || ENABLE_THERMAL_VALUE == 1, "ENABLE_THERMAL_VALUE must be 0 or 1.");
static_assert(THERMAL_FORCING_ENABLED_VALUE == 0 || THERMAL_FORCING_ENABLED_VALUE == 1, "THERMAL_FORCING_ENABLED_VALUE must be 0 or 1.");
static_assert(THERMAL_SOURCE_ENABLED_VALUE == 0 || THERMAL_SOURCE_ENABLED_VALUE == 1, "THERMAL_SOURCE_ENABLED_VALUE must be 0 or 1.");

static_assert(HYDRO_MOMENTS == 10, "Hydro MR layout expects 10 moments.");
static_assert(!ENABLE_THERMAL || NUM_MOMENTS == 14, "Thermal-enabled layout expects 14 moments.");
static_assert(ENABLE_THERMAL || NUM_MOMENTS == 10, "Thermal-disabled layout expects 10 moments.");

// =================================================================================================== //