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

constexpr natural_t L_CHAR = 256;
constexpr real_t U_CHAR = static_cast<real_t>(0.05);
constexpr real_t REYNOLDS = static_cast<real_t>(10000);
constexpr natural_t NSTEPS = 500000;
constexpr natural_t STAMP = 10000;

// =================================================================================================== //

constexpr real_t VISCOSITY = static_cast<real_t>((static_cast<double>(U_CHAR) * static_cast<double>(L_CHAR)) / static_cast<double>(REYNOLDS));
constexpr real_t TAU = static_cast<real_t>(0.5) + static_cast<real_t>(3.0) * VISCOSITY;
constexpr real_t OMEGA = static_cast<real_t>(static_cast<double>(1) / static_cast<double>(TAU));
constexpr real_t T_OMEGA = static_cast<real_t>(1.0) - OMEGA;
constexpr real_t OMEGA_D2 = static_cast<real_t>(0.5) * OMEGA;

// =================================================================================================== //

constexpr natural_t NX = L_CHAR;
constexpr natural_t NY = L_CHAR;
constexpr natural_t NZ = L_CHAR;
constexpr natural_t CELLS = NX * NY * NZ;
constexpr natural_t STRIDE = NX * NY;

// =================================================================================================== //

constexpr natural_t BLOCK_NX = 32;
constexpr natural_t BLOCK_NY = 4;
constexpr natural_t BLOCK_NZ = 4;
constexpr natural_t GRID_X = (NX + BLOCK_NX - 1) / BLOCK_NX;
constexpr natural_t GRID_Y = (NY + BLOCK_NY - 1) / BLOCK_NY;
constexpr natural_t GRID_Z = (NZ + BLOCK_NZ - 1) / BLOCK_NZ;

// =================================================================================================== //

constexpr natural_t NUM_MOMENTS = 10;

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

// =================================================================================================== //
