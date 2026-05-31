#pragma once

#include "constants.cuh"

constexpr uint8_t BULK = 0u;
constexpr uint8_t WEST = 1u << 0;
constexpr uint8_t EAST = 1u << 1;
constexpr uint8_t SOUTH = 1u << 2;
constexpr uint8_t NORTH = 1u << 3;
constexpr uint8_t BACK = 1u << 4;
constexpr uint8_t FRONT = 1u << 5;

// face nodes
constexpr uint8_t NORTH_FACE = NORTH;
constexpr uint8_t SOUTH_FACE = SOUTH;
constexpr uint8_t WEST_FACE = WEST;
constexpr uint8_t EAST_FACE = EAST;
constexpr uint8_t FRONT_FACE = FRONT;
constexpr uint8_t BACK_FACE = BACK;

// edge nodes
constexpr uint8_t NORTH_WEST = NORTH | WEST;
constexpr uint8_t NORTH_EAST = NORTH | EAST;
constexpr uint8_t NORTH_FRONT = NORTH | FRONT;
constexpr uint8_t NORTH_BACK = NORTH | BACK;
constexpr uint8_t SOUTH_WEST = SOUTH | WEST;
constexpr uint8_t SOUTH_EAST = SOUTH | EAST;
constexpr uint8_t SOUTH_FRONT = SOUTH | FRONT;
constexpr uint8_t SOUTH_BACK = SOUTH | BACK;
constexpr uint8_t WEST_FRONT = WEST | FRONT;
constexpr uint8_t WEST_BACK = WEST | BACK;
constexpr uint8_t EAST_FRONT = EAST | FRONT;
constexpr uint8_t EAST_BACK = EAST | BACK;

// corner nodes
constexpr uint8_t NORTH_WEST_FRONT = NORTH | WEST | FRONT;
constexpr uint8_t NORTH_WEST_BACK = NORTH | WEST | BACK;
constexpr uint8_t NORTH_EAST_FRONT = NORTH | EAST | FRONT;
constexpr uint8_t NORTH_EAST_BACK = NORTH | EAST | BACK;
constexpr uint8_t SOUTH_WEST_FRONT = SOUTH | WEST | FRONT;
constexpr uint8_t SOUTH_WEST_BACK = SOUTH | WEST | BACK;
constexpr uint8_t SOUTH_EAST_FRONT = SOUTH | EAST | FRONT;
constexpr uint8_t SOUTH_EAST_BACK = SOUTH | EAST | BACK;

__device__ [[nodiscard]] static inline uint8_t boundaryMask(
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    uint8_t type = BULK;

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