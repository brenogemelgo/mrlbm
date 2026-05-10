#pragma once

#include "deviceFunctions.cuh"
#include "bitmasks.cuh"

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

    real_t pop[27];

    constexpr_for<static_cast<natural_t>(0), static_cast<natural_t>(Q)>(
        [&](const auto qConst) noexcept
        {
            constexpr natural_t q = qConst();
            constexpr int cxi = CX[q];
            constexpr int cyi = CY[q];
            constexpr int czi = CZ[q];
            constexpr real_t cx = static_cast<real_t>(cxi);
            constexpr real_t cy = static_cast<real_t>(cyi);
            constexpr real_t cz = static_cast<real_t>(czi);
            constexpr real_t w = W[q];

            const int xs_i = static_cast<int>(x) - cxi;
            const int ys_i = static_cast<int>(y) - cyi;
            const int zs_i = static_cast<int>(z) - czi;

            if (xs_i >= 0 && xs_i < static_cast<int>(NX) && ys_i >= 0 && ys_i < static_cast<int>(NY) && zs_i >= 0 && zs_i < static_cast<int>(NZ))
            {
                const natural_t src = global3(static_cast<natural_t>(xs_i), static_cast<natural_t>(ys_i), static_cast<natural_t>(zs_i));

                const real_t rho_s = moments[src + CELLS * RHO];
                const real_t ux_s = moments[src + CELLS * UX];
                const real_t uy_s = moments[src + CELLS * UY];
                const real_t uz_s = moments[src + CELLS * UZ];
                const real_t mxx_s = moments[src + CELLS * MXX];
                const real_t myy_s = moments[src + CELLS * MYY];
                const real_t mzz_s = moments[src + CELLS * MZZ];
                const real_t mxy_s = moments[src + CELLS * MXY];
                const real_t mxz_s = moments[src + CELLS * MXZ];
                const real_t myz_s = moments[src + CELLS * MYZ];

                const real_t hxx = cx * cx - CS2;
                const real_t hyy = cy * cy - CS2;
                const real_t hzz = cz * cz - CS2;
                const real_t hxy = cx * cy;
                const real_t hxz = cx * cz;
                const real_t hyz = cy * cz;

                const real_t cu = cx * ux_s + cy * uy_s + cz * uz_s;

                const real_t mh = mxx_s * hxx + myy_s * hyy + mzz_s * hzz + mxy_s * hxy + mxz_s * hxz + myz_s * hyz;

                pop[q] = w * rho_s * (static_cast<real_t>(1.0) + cu + mh);
            }
            else
            {
                pop[q] = static_cast<real_t>(0.0);
            }
        });

    real_t rho = static_cast<real_t>(0);
    real_t ux = static_cast<real_t>(0);
    real_t uy = static_cast<real_t>(0);
    real_t uz = static_cast<real_t>(0);
    real_t mxx = static_cast<real_t>(0);
    real_t myy = static_cast<real_t>(0);
    real_t mzz = static_cast<real_t>(0);
    real_t mxy = static_cast<real_t>(0);
    real_t mxz = static_cast<real_t>(0);
    real_t myz = static_cast<real_t>(0);

    if (nodeType != BULK)
    {
#include "cavityBC.cuh"
    }
    else
    {
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

#undef STREAM_POP
