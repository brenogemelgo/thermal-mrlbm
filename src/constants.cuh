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

constexpr natural_t NX = 128;
constexpr natural_t NY = 128;
constexpr natural_t NZ = 512;

constexpr real_t REYNOLDS = static_cast<real_t>(1000);
constexpr real_t U_CHAR = static_cast<real_t>(0.0256);
constexpr real_t L_CHAR = static_cast<real_t>(24);

constexpr natural_t NSTEPS = 100000;
constexpr natural_t STAMP = 1000;

constexpr real_t BULK_VISCOSITY_FACTOR = static_cast<real_t>(50);

constexpr real_t PRANDTL = static_cast<real_t>(0.5);

constexpr real_t T_COLD = static_cast<real_t>(0);
constexpr real_t T_HOT = static_cast<real_t>(1);
constexpr real_t T_REF = static_cast<real_t>(0.5);

constexpr real_t BETA_THERMAL = static_cast<real_t>(0.1);
constexpr real_t THERMAL_SOURCE_VALUE = static_cast<real_t>(0);

constexpr real_t THERMAL_GRAVITY_X = static_cast<real_t>(0);
constexpr real_t THERMAL_GRAVITY_Y = static_cast<real_t>(0);
constexpr real_t THERMAL_GRAVITY_Z = static_cast<real_t>(1);

constexpr natural_t BLOCK_NX = 32;
constexpr natural_t BLOCK_NY = 4;
constexpr natural_t BLOCK_NZ = 4;

// =================================================================================================== //

constexpr natural_t CELLS = NX * NY * NZ;
constexpr natural_t STRIDE = NX * NY;
constexpr real_t RADIUS = static_cast<real_t>(0.5) * L_CHAR;

// =================================================================================================== //

constexpr real_t VISCOSITY = static_cast<real_t>((static_cast<double>(U_CHAR) * static_cast<double>(L_CHAR)) / static_cast<double>(REYNOLDS));

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

constexpr real_t THERMAL_FORCE_SCALE = static_cast<real_t>((static_cast<double>(U_CHAR) * static_cast<double>(U_CHAR)) / static_cast<double>(L_CHAR));
constexpr real_t THERMAL_DIFFUSIVITY = VISCOSITY / PRANDTL;
constexpr real_t TAU_THERMAL = static_cast<real_t>(0.5) + THERMAL_DIFFUSIVITY * static_cast<real_t>(3.0);
constexpr real_t OMEGA_THERMAL = static_cast<real_t>(static_cast<double>(1) / static_cast<double>(TAU_THERMAL));
constexpr real_t T_OMEGA_THERMAL = static_cast<real_t>(1.0) - OMEGA_THERMAL;

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
