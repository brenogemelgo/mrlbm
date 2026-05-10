#pragma once

// =================================================================================================== //

#include <cstdint>
#include <utility>
#include <cuda_fp16.h> // __half precision

// =================================================================================================== //

using natural_t = uint32_t;
using real_t = float;

// =================================================================================================== //

constexpr natural_t L_CHAR = 64;
constexpr real_t CHAR_VELOCITY = static_cast<real_t>(0.05);
constexpr real_t U_CHAR = CHAR_VELOCITY;
constexpr real_t REYNOLDS = static_cast<real_t>(3000);
constexpr natural_t NSTEPS = 100000;
constexpr natural_t STAMP = 1000;

// =================================================================================================== //

constexpr natural_t Q = 27;
constexpr int CX[Q] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1};
constexpr int CY[Q] = {0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1};
constexpr int CZ[Q] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1};
constexpr real_t W1 = static_cast<real_t>(static_cast<double>(8) / static_cast<double>(27));
constexpr real_t W2 = static_cast<real_t>(static_cast<double>(2) / static_cast<double>(27));
constexpr real_t W3 = static_cast<real_t>(static_cast<double>(1) / static_cast<double>(54));
constexpr real_t W4 = static_cast<real_t>(static_cast<double>(1) / static_cast<double>(216));
constexpr real_t W[Q] = {W1,
                         W2, W2, W2, W2, W2, W2,
                         W3, W3, W3, W3, W3, W3, W3, W3, W3, W3, W3, W3,
                         W4, W4, W4, W4, W4, W4, W4, W4};
constexpr real_t AS2 = static_cast<real_t>(3.0);
constexpr real_t AS4 = static_cast<real_t>(9.0);
constexpr real_t CS2 = static_cast<real_t>(static_cast<double>(1) / static_cast<double>(3));
constexpr real_t SCALE_I = static_cast<real_t>(3.0);
constexpr real_t SCALE_II = static_cast<real_t>(4.5);
constexpr real_t SCALE_IJ = static_cast<real_t>(9.0);
constexpr real_t EPS = static_cast<real_t>(1.0e-7);

// =================================================================================================== //

constexpr real_t VISCOSITY = static_cast<real_t>((static_cast<double>(CHAR_VELOCITY) * static_cast<double>(L_CHAR)) / static_cast<double>(REYNOLDS));
constexpr real_t TAU = static_cast<real_t>(0.5) + static_cast<real_t>(3.0) * VISCOSITY;
constexpr real_t OMEGA = static_cast<real_t>(static_cast<double>(1) / static_cast<double>(TAU));
constexpr real_t T_OMEGA = static_cast<real_t>(1.0) - OMEGA;
constexpr real_t OMEGA_D2 = static_cast<real_t>(0.5) * OMEGA;
constexpr real_t T_OMEGA_SCALE_II = T_OMEGA * SCALE_II;
constexpr real_t T_OMEGA_SCALE_IJ = T_OMEGA * SCALE_IJ;

// =================================================================================================== //

constexpr natural_t NX = L_CHAR;
constexpr natural_t NY = L_CHAR;
constexpr natural_t NZ = L_CHAR;
constexpr natural_t CELLS = NX * NY * NZ;

// =================================================================================================== //

constexpr natural_t BLOCK_NX = 8;
constexpr natural_t BLOCK_NY = 8;
constexpr natural_t BLOCK_NZ = 8;
constexpr natural_t NUM_BLOCK_X = (NX + BLOCK_NX - 1) / BLOCK_NX;
constexpr natural_t NUM_BLOCK_Y = (NY + BLOCK_NY - 1) / BLOCK_NY;
constexpr natural_t NUM_BLOCK_Z = (NZ + BLOCK_NZ - 1) / BLOCK_NZ;

// =================================================================================================== //

constexpr natural_t NUM_MOMENTS = 10;
constexpr natural_t NUM_FIELDS = NUM_MOMENTS;

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
