#pragma once

#include "deviceFunctions.cuh"
#include "bitmasks.cuh"

__global__ void streamCollide(
    const real_t *__restrict__ moments,
    real_t *__restrict__ dbuffer)
{
    const natural_t tx = threadIdx.x;
    const natural_t ty = threadIdx.y;
    const natural_t tz = threadIdx.z;
    const natural_t bx = blockIdx.x;
    const natural_t by = blockIdx.y;
    const natural_t bz = blockIdx.z;
    const natural_t x = bx * BLOCK_NX + tx;
    const natural_t y = by * BLOCK_NY + ty;
    const natural_t z = bz * BLOCK_NZ + tz;
    if (x >= NX || y >= NY || z >= NZ)
    {
        return;
    }
    const natural_t idx_ = idx(tx, ty, tz, bx, by, bz);

    // load moments
    real_t rho = moments[idx_ + CELLS * RHO];
    real_t ux = moments[idx_ + CELLS * UX];
    real_t uy = moments[idx_ + CELLS * UY];
    real_t uz = moments[idx_ + CELLS * UZ];
    real_t mxx = moments[idx_ + CELLS * MXX];
    real_t myy = moments[idx_ + CELLS * MYY];
    real_t mzz = moments[idx_ + CELLS * MZZ];
    real_t mxy = moments[idx_ + CELLS * MXY];
    real_t mxz = moments[idx_ + CELLS * MXZ];
    real_t myz = moments[idx_ + CELLS * MYZ];

    // check if boundary or interior
    const unsigned int nodeType = boundaryMask(x, y, z);
    if (nodeType != BULK)
    {
        // calculate moments at boundary
#include "cavityBC.cuh"
    }
    else
    {
        // calculate moments at interior
        rho = static_cast<real_t>(0);
        real_t jx = static_cast<real_t>(0);
        real_t jy = static_cast<real_t>(0);
        real_t jz = static_cast<real_t>(0);
        real_t hxxSum = static_cast<real_t>(0);
        real_t hyySum = static_cast<real_t>(0);
        real_t hzzSum = static_cast<real_t>(0);
        real_t hxySum = static_cast<real_t>(0);
        real_t hxzSum = static_cast<real_t>(0);
        real_t hyzSum = static_cast<real_t>(0);

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

                const natural_t xs = static_cast<natural_t>(static_cast<int>(x) - cxi);
                const natural_t ys = static_cast<natural_t>(static_cast<int>(y) - cyi);
                const natural_t zs = static_cast<natural_t>(static_cast<int>(z) - czi);
                const natural_t txs = xs % BLOCK_NX;
                const natural_t tys = ys % BLOCK_NY;
                const natural_t tzs = zs % BLOCK_NZ;
                const natural_t bxs = xs / BLOCK_NX;
                const natural_t bys = ys / BLOCK_NY;
                const natural_t bzs = zs / BLOCK_NZ;
                const natural_t src = idx(txs, tys, tzs, bxs, bys, bzs);

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

                const real_t fq = w * rho_s * (static_cast<real_t>(1) + cu + mh);

                rho += fq;
                jx += fq * cx;
                jy += fq * cy;
                jz += fq * cz;
                hxxSum += fq * hxx;
                hyySum += fq * hyy;
                hzzSum += fq * hzz;
                hxySum += fq * hxy;
                hxzSum += fq * hxz;
                hyzSum += fq * hyz;
            });

        const real_t invRho = static_cast<real_t>(1) / rho;

        ux = jx * invRho;
        uy = jy * invRho;
        uz = jz * invRho;
        mxx = hxxSum * invRho;
        myy = hyySum * invRho;
        mzz = hzzSum * invRho;
        mxy = hxySum * invRho;
        mxz = hxzSum * invRho;
        myz = hyzSum * invRho;
    }

    // scale
    ux = SCALE_I * ux;
    uy = SCALE_I * uy;
    uz = SCALE_I * uz;
    mxx = SCALE_II * mxx;
    myy = SCALE_II * myy;
    mzz = SCALE_II * mzz;
    mxy = SCALE_IJ * mxy;
    mxz = SCALE_IJ * mxz;
    myz = SCALE_IJ * myz;

    // collide
    mxx = T_OMEGA * mxx + OMEGA_D2 * ux * ux;
    myy = T_OMEGA * myy + OMEGA_D2 * uy * uy;
    mzz = T_OMEGA * mzz + OMEGA_D2 * uz * uz;
    mxy = T_OMEGA * mxy + OMEGA * ux * uy;
    mxz = T_OMEGA * mxz + OMEGA * ux * uz;
    myz = T_OMEGA * myz + OMEGA * uy * uz;

    // write to global memory
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
