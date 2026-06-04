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

// =================================================================================================== //

struct D3Q7;
struct D3Q19;
struct D3Q27;

using VelocitySet = D3Q27;
using ThermalVelocitySet = D3Q27;
using natural_t = uint32_t;
using real_t = float;

// =================================================================================================== //

#if defined(CASE_FOURIER)
#include "../cases/fourierAdvectionDiffusion.cuh"
using ActiveCase = FourierAdvectionDiffusionCase;
constexpr bool CASE_IS_FOURIER = true;
constexpr bool CASE_IS_JET = false;
#elif defined(CASE_SLOT)
#include "../cases/verticalBuoyantSlot.cuh"
using ActiveCase = VerticalBuoyantSlotCase;
constexpr bool CASE_IS_FOURIER = false;
constexpr bool CASE_IS_JET = false;
#elif defined(CASE_JET)
#include "../cases/thermalJet.cuh"
using ActiveCase = ThermalJetCase;
constexpr bool CASE_IS_FOURIER = false;
constexpr bool CASE_IS_JET = true;
#else
#error "No case selected. Compile with -DCASE_FOURIER, -DCASE_SLOT, or -DCASE_JET."
#endif

// =================================================================================================== //

constexpr natural_t NX = ActiveCase::NX;
constexpr natural_t NY = ActiveCase::NY;
constexpr natural_t NZ = ActiveCase::NZ;

constexpr real_t REYNOLDS = ActiveCase::RE;
constexpr real_t U_CHAR = ActiveCase::U_CHAR;
constexpr real_t L_CHAR = ActiveCase::L_CHAR;

constexpr natural_t NSTEPS = ActiveCase::NSTEPS;
constexpr natural_t STAMP = ActiveCase::STAMP;

constexpr real_t BULK_VISCOSITY_FACTOR = static_cast<real_t>(1);

constexpr real_t PRANDTL = ActiveCase::PR;

constexpr real_t T_COLD = ActiveCase::TCOLD;
constexpr real_t T_HOT = ActiveCase::THOT;
constexpr real_t T_REF = ActiveCase::T_REF;

constexpr real_t BETA_THERMAL = ActiveCase::GBETA;
constexpr real_t THERMAL_SOURCE_VALUE = static_cast<real_t>(0);

constexpr real_t THERMAL_GRAVITY_X = ActiveCase::GRAVITY_X;
constexpr real_t THERMAL_GRAVITY_Y = ActiveCase::GRAVITY_Y;
constexpr real_t THERMAL_GRAVITY_Z = ActiveCase::GRAVITY_Z;

constexpr bool ENABLE_BUOYANCY = ActiveCase::ENABLE_BUOYANCY;
constexpr bool ENABLE_INLET = ActiveCase::ENABLE_INLET;
constexpr bool ENABLE_WALLS = ActiveCase::ENABLE_WALLS;
constexpr bool ENABLE_OUTFLOW = ActiveCase::ENABLE_OUTFLOW;

constexpr bool PERIODIC_X = ActiveCase::PERIODIC_X;
constexpr bool PERIODIC_Y = ActiveCase::PERIODIC_Y;
constexpr bool PERIODIC_Z = ActiveCase::PERIODIC_Z;

constexpr bool ENABLE_SMOOTH_JET_PROFILE = false;
constexpr real_t JET_AMBIENT_UZ = static_cast<real_t>(0);
constexpr real_t JET_INLET_SHEAR_THICKNESS_CELLS = static_cast<real_t>(2);
constexpr real_t JET_INLET_HALO_THICKNESS_CELLS = static_cast<real_t>(2) * JET_INLET_SHEAR_THICKNESS_CELLS;
constexpr bool SMOOTH_INLET_TEMPERATURE = false;

constexpr bool ENABLE_SYNTHETIC_TURBULENT_INFLOW = CASE_IS_JET;

constexpr natural_t TURBULENT_INFLOW_MODES = 16;
constexpr real_t TURBULENT_INFLOW_INTENSITY_RADIAL = static_cast<real_t>(0.04);
constexpr real_t TURBULENT_INFLOW_INTENSITY_AZIMUTHAL = static_cast<real_t>(0.04);
constexpr real_t TURBULENT_INFLOW_INTENSITY_AXIAL = static_cast<real_t>(0.025);

constexpr real_t TURBULENT_INFLOW_LMIN_CELLS = static_cast<real_t>(3);
constexpr real_t TURBULENT_INFLOW_LMAX_DIAM = static_cast<real_t>(0.75);
constexpr real_t TURBULENT_INFLOW_CONVECTION_SPEED = U_CHAR;

constexpr natural_t TURBULENT_INFLOW_ENVELOPE_MODE = 0;
constexpr real_t TURBULENT_INFLOW_MAX_PERTURBATION = static_cast<real_t>(0.35) * U_CHAR;

constexpr natural_t TURBULENT_INFLOW_SEED = 1337u;

constexpr natural_t BLOCK_NX = 32;
constexpr natural_t BLOCK_NY = 4;
constexpr natural_t BLOCK_NZ = 4;

// =================================================================================================== //

constexpr natural_t CELLS = NX * NY * NZ;
constexpr natural_t STRIDE = NX * NY;
constexpr real_t RADIUS = static_cast<real_t>(0.5) * L_CHAR;

// =================================================================================================== //

constexpr real_t VISCOSITY = ActiveCase::NU;

constexpr real_t TAU_SHEAR = static_cast<real_t>(0.5) + static_cast<real_t>(3.0) * VISCOSITY;
constexpr real_t OMEGA_SHEAR = static_cast<real_t>(static_cast<double>(1) / static_cast<double>(TAU_SHEAR));
constexpr real_t T_OMEGA_SHEAR = static_cast<real_t>(1.0) - OMEGA_SHEAR;
constexpr real_t OMEGA_D2_SHEAR = static_cast<real_t>(0.5) * OMEGA_SHEAR;
constexpr real_t TT_OMEGA_SHEAR = static_cast<real_t>(1.0) - OMEGA_D2_SHEAR;

constexpr real_t TAU_BULK = static_cast<real_t>(0.5) + BULK_VISCOSITY_FACTOR * (TAU_SHEAR - static_cast<real_t>(0.5));
constexpr real_t OMEGA_BULK = static_cast<real_t>(static_cast<double>(1) / static_cast<double>(TAU_BULK));
constexpr real_t T_OMEGA_BULK = static_cast<real_t>(1.0) - OMEGA_BULK;
constexpr real_t OMEGA_D2_BULK = static_cast<real_t>(0.5) * OMEGA_BULK;
constexpr real_t TT_OMEGA_BULK = static_cast<real_t>(1.0) - OMEGA_D2_BULK;

// =================================================================================================== //

constexpr real_t THERMAL_DIFFUSIVITY = ActiveCase::ALPHA;
constexpr real_t TAU_THERMAL = static_cast<real_t>(0.5) + THERMAL_DIFFUSIVITY * static_cast<real_t>(3.0);
constexpr real_t OMEGA_THERMAL = static_cast<real_t>(static_cast<double>(1) / static_cast<double>(TAU_THERMAL));
constexpr real_t T_OMEGA_THERMAL = static_cast<real_t>(1.0) - OMEGA_THERMAL;

// =================================================================================================== //

constexpr real_t SPONGE_GAIN = static_cast<real_t>(100);
constexpr natural_t SPONGE_CELLS = (NZ / 12u) > 0u ? (NZ / 12u) : 1u;
constexpr real_t SPONGE_INV_NZ_M1 = static_cast<real_t>(static_cast<double>(1) / static_cast<double>(NZ - 1u));
constexpr real_t SPONGE_WIDTH = static_cast<real_t>(static_cast<double>(SPONGE_CELLS) / static_cast<double>(NZ - 1u));
constexpr real_t SPONGE_Z_START = static_cast<real_t>(static_cast<double>(NZ - 1u - SPONGE_CELLS) / static_cast<double>(NZ - 1u));
constexpr real_t SPONGE_INV_WIDTH = static_cast<real_t>(static_cast<double>(1) / static_cast<double>(SPONGE_WIDTH));

constexpr real_t TAU_SHEAR_SPONGE_MAX = static_cast<real_t>(0.5) + (TAU_SHEAR - static_cast<real_t>(0.5)) * (static_cast<real_t>(1) + SPONGE_GAIN);
constexpr real_t TAU_BULK_SPONGE_MAX = static_cast<real_t>(0.5) + (TAU_BULK - static_cast<real_t>(0.5)) * (static_cast<real_t>(1) + SPONGE_GAIN);
constexpr real_t TAU_THERMAL_SPONGE_MAX = static_cast<real_t>(0.5) + (TAU_THERMAL - static_cast<real_t>(0.5)) * (static_cast<real_t>(1) + SPONGE_GAIN);

// =================================================================================================== //

constexpr natural_t GRID_X = (NX + BLOCK_NX - 1) / BLOCK_NX;
constexpr natural_t GRID_Y = (NY + BLOCK_NY - 1) / BLOCK_NY;
constexpr natural_t GRID_Z = (NZ + BLOCK_NZ - 1) / BLOCK_NZ;

// =================================================================================================== //

constexpr natural_t HYDRO_MOMENTS = 10;
constexpr natural_t THERMAL_MOMENTS = 4;
constexpr natural_t NUM_MOMENTS = HYDRO_MOMENTS + THERMAL_MOMENTS;

// =================================================================================================== //

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
constexpr natural_t TUX = 11;
constexpr natural_t TUY = 12;
constexpr natural_t TUZ = 13;

// =================================================================================================== //
