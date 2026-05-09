#pragma once

#include "deviceFunctions.cuh"

__global__ void cavityInit(real_t *moments)
{
    const natural_t x = threadIdx.x + BLOCK_NX * blockIdx.x;
    const natural_t y = threadIdx.y + BLOCK_NY * blockIdx.y;
    const natural_t z = threadIdx.z + BLOCK_NZ * blockIdx.z;

    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    const natural_t id = cellIdx(x, y, z);
    const real_t ux = static_cast<real_t>(0);
    const real_t uy = static_cast<real_t>(0);
    const real_t uz = (z == NZ - 1) ? CHAR_VELOCITY : static_cast<real_t>(0);

    moment(moments, RHO, id) = static_cast<real_t>(1);
    moment(moments, UX, id) = ux;
    moment(moments, UY, id) = uy;
    moment(moments, UZ, id) = uz;
    moment(moments, MXX, id) = ux * ux;
    moment(moments, MYY, id) = uy * uy;
    moment(moments, MZZ, id) = uz * uz;
    moment(moments, MXY, id) = ux * uy;
    moment(moments, MXZ, id) = ux * uz;
    moment(moments, MYZ, id) = uy * uz;

    moment(moments, NUM_MOMENTS + RHO, id) = static_cast<real_t>(1);
    moment(moments, NUM_MOMENTS + UX, id) = ux;
    moment(moments, NUM_MOMENTS + UY, id) = uy;
    moment(moments, NUM_MOMENTS + UZ, id) = uz;
    moment(moments, NUM_MOMENTS + MXX, id) = ux * ux;
    moment(moments, NUM_MOMENTS + MYY, id) = uy * uy;
    moment(moments, NUM_MOMENTS + MZZ, id) = uz * uz;
    moment(moments, NUM_MOMENTS + MXY, id) = ux * uy;
    moment(moments, NUM_MOMENTS + MXZ, id) = ux * uz;
    moment(moments, NUM_MOMENTS + MYZ, id) = uy * uz;
}
