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

    const natural_t idx = global3(x, y, z);
    const uint8_t nodeType = boundaryMask(x, y, z);

    real_t pop[VelocitySet::Q()];

    real_t rho;
    real_t ux, uy, uz;
    real_t mxx, myy, mzz, mxy, mxz, myz;

    constexpr_for<0, VelocitySet::Q()>(
        [&](const auto Q) noexcept
        {
            const int xs = static_cast<int>(x) - VelocitySet::cx<Q>();
            const int ys = static_cast<int>(y) - VelocitySet::cy<Q>();
            const int zs = static_cast<int>(z) - VelocitySet::cz<Q>();

            const bool validSrc = xs >= 0 && xs < static_cast<int>(NX) && ys >= 0 && ys < static_cast<int>(NY) && zs >= 0 && zs < static_cast<int>(NZ);

            if (validSrc)
            {
                const natural_t src = global3(static_cast<natural_t>(xs),
                                              static_cast<natural_t>(ys),
                                              static_cast<natural_t>(zs));

                const real_t cu = static_cast<real_t>(VelocitySet::cx<Q>()) * moments[midx(src, UX)] +
                                  static_cast<real_t>(VelocitySet::cy<Q>()) * moments[midx(src, UY)] +
                                  static_cast<real_t>(VelocitySet::cz<Q>()) * moments[midx(src, UZ)];

                const real_t mh = moments[midx(src, MXX)] * VelocitySet::hxx<Q>() +
                                  moments[midx(src, MYY)] * VelocitySet::hyy<Q>() +
                                  moments[midx(src, MZZ)] * VelocitySet::hzz<Q>() +
                                  moments[midx(src, MXY)] * VelocitySet::hxy<Q>() +
                                  moments[midx(src, MXZ)] * VelocitySet::hxz<Q>() +
                                  moments[midx(src, MYZ)] * VelocitySet::hyz<Q>();

                pop[Q] = VelocitySet::w<Q>() * moments[midx(src, RHO)] * (static_cast<real_t>(1) + cu + mh);
            }
            else
            {
                pop[Q] = static_cast<real_t>(0);
            }
        });

    if (nodeType != BULK)
    {
        dispatchIRBCBoundary(pop, nodeType, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
    }
    else
    {
        rho = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
        const real_t invRho = static_cast<real_t>(1) / rho;

        ux = (pop[1] - pop[2] + pop[7] - pop[8] + pop[9] - pop[10] + pop[13] - pop[14] + pop[15] - pop[16] + pop[19] - pop[20] + pop[21] - pop[22] + pop[23] - pop[24] - pop[25] + pop[26]) * invRho;
        uy = (pop[3] - pop[4] + pop[7] - pop[8] + pop[11] - pop[12] - pop[13] + pop[14] + pop[17] - pop[18] + pop[19] - pop[20] + pop[21] - pop[22] - pop[23] + pop[24] + pop[25] - pop[26]) * invRho;
        uz = (pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] - pop[15] + pop[16] - pop[17] + pop[18] + pop[19] - pop[20] - pop[21] + pop[22] + pop[23] - pop[24] + pop[25] - pop[26]) * invRho;

        mxx = (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]) * invRho - VelocitySet::cs2();
        myy = (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]) * invRho - VelocitySet::cs2();
        mzz = (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]) * invRho - VelocitySet::cs2();
        mxy = (pop[7] + pop[8] + pop[19] + pop[20] + pop[21] + pop[22] - pop[13] - pop[14] - pop[23] - pop[24] - pop[25] - pop[26]) * invRho;
        mxz = (pop[9] + pop[10] + pop[19] + pop[20] + pop[23] + pop[24] - pop[15] - pop[16] - pop[21] - pop[22] - pop[25] - pop[26]) * invRho;
        myz = (pop[11] + pop[12] + pop[19] + pop[20] + pop[25] + pop[26] - pop[17] - pop[18] - pop[21] - pop[22] - pop[23] - pop[24]) * invRho;
    }

    ux *= VelocitySet::scaleI();
    uy *= VelocitySet::scaleI();
    uz *= VelocitySet::scaleI();
    mxx *= VelocitySet::scaleII();
    myy *= VelocitySet::scaleII();
    mzz *= VelocitySet::scaleII();
    mxy *= VelocitySet::scaleIJ();
    mxz *= VelocitySet::scaleIJ();
    myz *= VelocitySet::scaleIJ();

    mxx = T_OMEGA * mxx + OMEGA_D2 * ux * ux;
    myy = T_OMEGA * myy + OMEGA_D2 * uy * uy;
    mzz = T_OMEGA * mzz + OMEGA_D2 * uz * uz;
    mxy = T_OMEGA * mxy + OMEGA * ux * uy;
    mxz = T_OMEGA * mxz + OMEGA * ux * uz;
    myz = T_OMEGA * myz + OMEGA * uy * uz;

    dbuffer[midx(idx, RHO)] = rho;
    dbuffer[midx(idx, UX)] = ux;
    dbuffer[midx(idx, UY)] = uy;
    dbuffer[midx(idx, UZ)] = uz;
    dbuffer[midx(idx, MXX)] = mxx;
    dbuffer[midx(idx, MYY)] = myy;
    dbuffer[midx(idx, MZZ)] = mzz;
    dbuffer[midx(idx, MXY)] = mxy;
    dbuffer[midx(idx, MXZ)] = mxz;
    dbuffer[midx(idx, MYZ)] = myz;
}