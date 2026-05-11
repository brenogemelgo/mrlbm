#pragma once

#include "deviceFunctions.cuh"
#include "irbcBoundary.cuh"

__global__ void streamCollide(
    const real_t *__restrict__ moments,
    real_t *__restrict__ dbuffer)
{
    const natural_t x = blockIdx.x * BLOCK_NX + threadIdx.x;
    const natural_t y = blockIdx.y * BLOCK_NY + threadIdx.y;
    const natural_t z = blockIdx.z * BLOCK_NZ + threadIdx.z;

    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }

    const natural_t idx_ = global3(x, y, z);
    const unsigned int nodeType = boundaryMask(x, y, z);

    real_t rho;
    real_t ux;
    real_t uy;
    real_t uz;
    real_t mxx;
    real_t myy;
    real_t mzz;
    real_t mxy;
    real_t mxz;
    real_t myz;

    if (nodeType != BULK)
    {
        dispatchIRBCBoundary(moments, x, y, z, nodeType, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
    }
    else
    {
        real_t pop[27];

        constexpr_for<static_cast<natural_t>(0), static_cast<natural_t>(Q)>(
            [&](const auto qConst) noexcept
            {
                constexpr natural_t q = qConst();
                pop[q] = reconstructStreamedPopulation<q>(moments, x, y, z);
            });

        rho = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
        const real_t invRho = static_cast<real_t>(1) / rho;

        ux = (pop[1] - pop[2] + pop[7] - pop[8] + pop[9] - pop[10] + pop[13] - pop[14] + pop[15] - pop[16] + pop[19] - pop[20] + pop[21] - pop[22] + pop[23] - pop[24] - pop[25] + pop[26]) * invRho;
        uy = (pop[3] - pop[4] + pop[7] - pop[8] + pop[11] - pop[12] - pop[13] + pop[14] + pop[17] - pop[18] + pop[19] - pop[20] + pop[21] - pop[22] - pop[23] + pop[24] + pop[25] - pop[26]) * invRho;
        uz = (pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] - pop[15] + pop[16] - pop[17] + pop[18] + pop[19] - pop[20] - pop[21] + pop[22] + pop[23] - pop[24] + pop[25] - pop[26]) * invRho;

        mxx = (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]) * invRho - CS2;
        myy = (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]) * invRho - CS2;
        mzz = (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]) * invRho - CS2;
        mxy = (pop[7] + pop[8] + pop[19] + pop[20] + pop[21] + pop[22] - pop[13] - pop[14] - pop[23] - pop[24] - pop[25] - pop[26]) * invRho;
        mxz = (pop[9] + pop[10] + pop[19] + pop[20] + pop[23] + pop[24] - pop[15] - pop[16] - pop[21] - pop[22] - pop[25] - pop[26]) * invRho;
        myz = (pop[11] + pop[12] + pop[19] + pop[20] + pop[25] + pop[26] - pop[17] - pop[18] - pop[21] - pop[22] - pop[23] - pop[24]) * invRho;
    }

    ux = SCALE_I * ux;
    uy = SCALE_I * uy;
    uz = SCALE_I * uz;
    mxx = SCALE_II * mxx;
    myy = SCALE_II * myy;
    mzz = SCALE_II * mzz;
    mxy = SCALE_IJ * mxy;
    mxz = SCALE_IJ * mxz;
    myz = SCALE_IJ * myz;

    mxx = T_OMEGA * mxx + OMEGA_D2 * ux * ux;
    myy = T_OMEGA * myy + OMEGA_D2 * uy * uy;
    mzz = T_OMEGA * mzz + OMEGA_D2 * uz * uz;
    mxy = T_OMEGA * mxy + OMEGA * ux * uy;
    mxz = T_OMEGA * mxz + OMEGA * ux * uz;
    myz = T_OMEGA * myz + OMEGA * uy * uz;

    dbuffer[idx_ + CELLS * RHO] = rho;
    dbuffer[idx_ + CELLS * UX] = ux;
    dbuffer[idx_ + CELLS * UY] = uy;
    dbuffer[idx_ + CELLS * UZ] = uz;
    dbuffer[idx_ + CELLS * MXX] = mxx;
    dbuffer[idx_ + CELLS * MYY] = myy;
    dbuffer[idx_ + CELLS * MZZ] = mzz;
    dbuffer[idx_ + CELLS * MXY] = mxy;
    dbuffer[idx_ + CELLS * MXZ] = mxz;
    dbuffer[idx_ + CELLS * MYZ] = myz;
}
