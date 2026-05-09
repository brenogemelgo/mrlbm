#pragma once

#include <cstdint>
#include <utility>

using natural_t = uint32_t;
using real_t = float;

constexpr natural_t CAVITY_LENGTH = 64;
constexpr natural_t NX = CAVITY_LENGTH;
constexpr natural_t NY = CAVITY_LENGTH;
constexpr natural_t NZ = CAVITY_LENGTH;
constexpr natural_t STRIDE = NX * NY;
constexpr natural_t CELLS = STRIDE * NZ;

constexpr natural_t BLOCK_NX = 8;
constexpr natural_t BLOCK_NY = 8;
constexpr natural_t BLOCK_NZ = 8;
constexpr natural_t NUM_BLOCK_X = (NX + BLOCK_NX - 1) / BLOCK_NX;
constexpr natural_t NUM_BLOCK_Y = (NY + BLOCK_NY - 1) / BLOCK_NY;
constexpr natural_t NUM_BLOCK_Z = (NZ + BLOCK_NZ - 1) / BLOCK_NZ;

constexpr natural_t Q = 27;
constexpr int CX[Q] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1};
constexpr int CY[Q] = {0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1};
constexpr int CZ[Q] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1};
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
constexpr real_t CS2 = static_cast<real_t>(static_cast<double>(1.0) / static_cast<double>(3.0));

constexpr real_t REYNOLDS = static_cast<real_t>(100);
constexpr real_t CHAR_LENGTH = static_cast<real_t>(CAVITY_LENGTH - 1);
constexpr real_t CHAR_VELOCITY = static_cast<real_t>(0.05);
constexpr real_t VISCOSITY = static_cast<real_t>((static_cast<double>(CHAR_VELOCITY) * static_cast<double>(CHAR_LENGTH)) / static_cast<double>(REYNOLDS));
constexpr real_t TAU = static_cast<real_t>(0.5) + static_cast<real_t>(3.0) * VISCOSITY;
constexpr real_t OMEGA = static_cast<real_t>(static_cast<double>(1) / static_cast<double>(TAU));

constexpr natural_t NSTEPS = 10000;
constexpr natural_t STAMP = 1000;

constexpr natural_t NUM_MOMENTS = 10;
constexpr natural_t NUM_BUFFERS = 2;
constexpr natural_t NUM_FIELDS = NUM_MOMENTS * NUM_BUFFERS;

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
