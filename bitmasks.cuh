#pragma once

#include "constants.cuh"

constexpr unsigned int BULK = 0u;
constexpr unsigned int WEST = 1u << 0;
constexpr unsigned int EAST = 1u << 1;
constexpr unsigned int SOUTH = 1u << 2;
constexpr unsigned int NORTH = 1u << 3;
constexpr unsigned int BACK = 1u << 4;
constexpr unsigned int FRONT = 1u << 5;

// face nodes
constexpr unsigned int NORTH_FACE = NORTH;
constexpr unsigned int SOUTH_FACE = SOUTH;
constexpr unsigned int WEST_FACE = WEST;
constexpr unsigned int EAST_FACE = EAST;
constexpr unsigned int FRONT_FACE = FRONT;
constexpr unsigned int BACK_FACE = BACK;

// edge nodes
constexpr unsigned int NORTH_WEST = NORTH | WEST;
constexpr unsigned int NORTH_EAST = NORTH | EAST;
constexpr unsigned int NORTH_FRONT = NORTH | FRONT;
constexpr unsigned int NORTH_BACK = NORTH | BACK;
constexpr unsigned int SOUTH_WEST = SOUTH | WEST;
constexpr unsigned int SOUTH_EAST = SOUTH | EAST;
constexpr unsigned int SOUTH_FRONT = SOUTH | FRONT;
constexpr unsigned int SOUTH_BACK = SOUTH | BACK;
constexpr unsigned int WEST_FRONT = WEST | FRONT;
constexpr unsigned int WEST_BACK = WEST | BACK;
constexpr unsigned int EAST_FRONT = EAST | FRONT;
constexpr unsigned int EAST_BACK = EAST | BACK;

// corner nodes
constexpr unsigned int NORTH_WEST_FRONT = NORTH | WEST | FRONT;
constexpr unsigned int NORTH_WEST_BACK = NORTH | WEST | BACK;
constexpr unsigned int NORTH_EAST_FRONT = NORTH | EAST | FRONT;
constexpr unsigned int NORTH_EAST_BACK = NORTH | EAST | BACK;
constexpr unsigned int SOUTH_WEST_FRONT = SOUTH | WEST | FRONT;
constexpr unsigned int SOUTH_WEST_BACK = SOUTH | WEST | BACK;
constexpr unsigned int SOUTH_EAST_FRONT = SOUTH | EAST | FRONT;
constexpr unsigned int SOUTH_EAST_BACK = SOUTH | EAST | BACK;

__device__ [[nodiscard]] static inline unsigned int boundaryMask(
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    unsigned int type = BULK;

    if (x == 0)
    {
        type |= WEST;
    }
    if (x == NX - 1)
    {
        type |= EAST;
    }
    if (y == 0)
    {
        type |= SOUTH;
    }
    if (y == NY - 1)
    {
        type |= NORTH;
    }
    if (z == 0)
    {
        type |= BACK;
    }
    if (z == NZ - 1)
    {
        type |= FRONT;
    }

    return type;
}