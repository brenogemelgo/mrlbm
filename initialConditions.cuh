#pragma once

#include "deviceFunctions.cuh"

__global__ void cavityInit(
    real_t *moments,
    real_t *dbuffer)
{
    const natural_t x = threadIdx.x + BLOCK_NX * blockIdx.x;
    const natural_t y = threadIdx.y + BLOCK_NY * blockIdx.y;
    const natural_t z = threadIdx.z + BLOCK_NZ * blockIdx.z;

    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    const natural_t id = idx(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);

    moment(moments, RHO, id) = static_cast<real_t>(1);
    moment(moments, UX, id) = static_cast<real_t>(0);
    moment(moments, UY, id) = static_cast<real_t>(0);
    moment(moments, UZ, id) = static_cast<real_t>(0);
    moment(moments, MXX, id) = static_cast<real_t>(0);
    moment(moments, MYY, id) = static_cast<real_t>(0);
    moment(moments, MZZ, id) = static_cast<real_t>(0);
    moment(moments, MXY, id) = static_cast<real_t>(0);
    moment(moments, MXZ, id) = static_cast<real_t>(0);
    moment(moments, MYZ, id) = static_cast<real_t>(0);

    moment(dbuffer, RHO, id) = static_cast<real_t>(1);
    moment(dbuffer, UX, id) = static_cast<real_t>(0);
    moment(dbuffer, UY, id) = static_cast<real_t>(0);
    moment(dbuffer, UZ, id) = static_cast<real_t>(0);
    moment(dbuffer, MXX, id) = static_cast<real_t>(0);
    moment(dbuffer, MYY, id) = static_cast<real_t>(0);
    moment(dbuffer, MZZ, id) = static_cast<real_t>(0);
    moment(dbuffer, MXY, id) = static_cast<real_t>(0);
    moment(dbuffer, MXZ, id) = static_cast<real_t>(0);
    moment(dbuffer, MYZ, id) = static_cast<real_t>(0);
}
