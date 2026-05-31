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

#define NX_VALUE 128
#define NY_VALUE 128
#define NZ_VALUE 512

#define REYNOLDS_VALUE 1000
#define JET_VELOCITY_VALUE 0.05
#define JET_DIAMETER_VALUE 24

#define NSTEPS_VALUE 100000
#define STAMP_VALUE 1000

#define BULK_RELAXATION_FACTOR_VALUE 0.75

#define PRANDTL_VALUE 0.71

#define T_COLD_VALUE 0.0
#define T_HOT_VALUE 1.0
#define T_REF_VALUE 0.5

#define BETA_THERMAL_VALUE 0.1
#define THERMAL_SOURCE_VALUE_PARAM 0.0

#define THERMAL_GRAVITY_X_VALUE 0.0
#define THERMAL_GRAVITY_Y_VALUE 0.0
#define THERMAL_GRAVITY_Z_VALUE 1.0

#define THERMAL_RHO0_VALUE 1.0
#define THERMAL_CS2_VALUE 0.25

#define JET_UX_VALUE 0.0
#define JET_UY_VALUE 0.0
#define JET_UZ_VALUE JET_VELOCITY_VALUE

#define JET_TEMPERATURE_VALUE T_HOT_VALUE
#define AMBIENT_TEMPERATURE_VALUE T_COLD_VALUE

#define BLOCK_NX_VALUE 32
#define BLOCK_NY_VALUE 4
#define BLOCK_NZ_VALUE 4

// =================================================================================================== //
// Domain

constexpr natural_t NX = static_cast<natural_t>(NX_VALUE);
constexpr natural_t NY = static_cast<natural_t>(NY_VALUE);
constexpr natural_t NZ = static_cast<natural_t>(NZ_VALUE);

constexpr natural_t CELLS = NX * NY * NZ;
constexpr natural_t STRIDE = NX * NY;

// =================================================================================================== //
// Jet parameters

constexpr real_t REYNOLDS = static_cast<real_t>(REYNOLDS_VALUE);

constexpr real_t JET_VELOCITY = static_cast<real_t>(JET_VELOCITY_VALUE);
constexpr real_t JET_DIAMETER = static_cast<real_t>(JET_DIAMETER_VALUE);
constexpr real_t JET_RADIUS = static_cast<real_t>(0.5) * JET_DIAMETER;

constexpr real_t JET_UX = static_cast<real_t>(JET_UX_VALUE);
constexpr real_t JET_UY = static_cast<real_t>(JET_UY_VALUE);
constexpr real_t JET_UZ = static_cast<real_t>(JET_UZ_VALUE);

constexpr real_t JET_TEMPERATURE = static_cast<real_t>(JET_TEMPERATURE_VALUE);
constexpr real_t AMBIENT_TEMPERATURE = static_cast<real_t>(AMBIENT_TEMPERATURE_VALUE);

constexpr natural_t NSTEPS = static_cast<natural_t>(NSTEPS_VALUE);
constexpr natural_t STAMP = static_cast<natural_t>(STAMP_VALUE);

constexpr real_t U_CHAR = JET_VELOCITY;

// =================================================================================================== //
// Hydrodynamic relaxation

constexpr real_t VISCOSITY = static_cast<real_t>(
    (static_cast<double>(JET_VELOCITY) * static_cast<double>(JET_DIAMETER)) /
    static_cast<double>(REYNOLDS));

constexpr real_t TAU_NU = static_cast<real_t>(0.5) + static_cast<real_t>(3.0) * VISCOSITY;

constexpr real_t TAU = TAU_NU;
constexpr real_t OMEGA = static_cast<real_t>(static_cast<double>(1) / static_cast<double>(TAU));
constexpr real_t T_OMEGA = static_cast<real_t>(1.0) - OMEGA;
constexpr real_t OMEGA_D2 = static_cast<real_t>(0.5) * OMEGA;

constexpr real_t BULK_RELAXATION_FACTOR = static_cast<real_t>(BULK_RELAXATION_FACTOR_VALUE);

constexpr real_t TAU_BULK =
    static_cast<real_t>(0.5) +
    BULK_RELAXATION_FACTOR * (TAU_NU - static_cast<real_t>(0.5));

constexpr real_t OMEGA_BULK = static_cast<real_t>(
    static_cast<double>(1) / static_cast<double>(TAU_BULK));

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

constexpr real_t THERMAL_FORCE_SCALE = static_cast<real_t>(
    (static_cast<double>(JET_VELOCITY) * static_cast<double>(JET_VELOCITY)) /
    static_cast<double>(JET_DIAMETER));

constexpr real_t THERMAL_DIFFUSIVITY = VISCOSITY / PRANDTL;

constexpr real_t TAU_T =
    static_cast<real_t>(0.5) + THERMAL_DIFFUSIVITY / THERMAL_CS2;

constexpr real_t OMEGA_T = static_cast<real_t>(
    static_cast<double>(1) / static_cast<double>(TAU_T));

constexpr real_t T_OMEGA_T = static_cast<real_t>(1.0) - OMEGA_T;

// =================================================================================================== //
// Derived jet diagnostics

constexpr real_t RE_JET = JET_UZ * JET_DIAMETER / VISCOSITY;

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
// Sanity checks

static_assert(NX > 2, "NX must be larger than 2.");
static_assert(NY > 2, "NY must be larger than 2.");
static_assert(NZ > 2, "NZ must be larger than 2.");

static_assert(REYNOLDS > static_cast<real_t>(0), "REYNOLDS must be positive.");
static_assert(JET_VELOCITY > static_cast<real_t>(0), "JET_VELOCITY must be positive.");
static_assert(JET_DIAMETER > static_cast<real_t>(0), "JET_DIAMETER must be positive.");
static_assert(JET_DIAMETER < static_cast<real_t>(NX), "JET_DIAMETER must be smaller than NX.");
static_assert(JET_DIAMETER < static_cast<real_t>(NY), "JET_DIAMETER must be smaller than NY.");

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

constexpr bool THERMAL_SOURCE_ACTIVE =
    THERMAL_SOURCE_VALUE != static_cast<real_t>(0);

constexpr bool THERMAL_FORCE_X_ACTIVE =
    BETA_THERMAL != static_cast<real_t>(0) &&
    THERMAL_GRAVITY_X != static_cast<real_t>(0);

constexpr bool THERMAL_FORCE_Y_ACTIVE =
    BETA_THERMAL != static_cast<real_t>(0) &&
    THERMAL_GRAVITY_Y != static_cast<real_t>(0);

constexpr bool THERMAL_FORCE_Z_ACTIVE =
    BETA_THERMAL != static_cast<real_t>(0) &&
    THERMAL_GRAVITY_Z != static_cast<real_t>(0);

constexpr bool THERMAL_FORCE_ACTIVE =
    THERMAL_FORCE_X_ACTIVE ||
    THERMAL_FORCE_Y_ACTIVE ||
    THERMAL_FORCE_Z_ACTIVE;