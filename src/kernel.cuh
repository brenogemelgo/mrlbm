#pragma once

#include "deviceFunctions.cuh"
#include "boundary/deviceRuntime.cuh"

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
        dispatchIRBCBoundary(moments, x, y, z, nodeType, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
    }
    else
    {
        constexpr_for<0, VelocitySet::Q()>(
            [&](const auto Q) noexcept
            {
                constexpr int cx = VelocitySet::cx<Q>();
                constexpr int cy = VelocitySet::cy<Q>();
                constexpr int cz = VelocitySet::cz<Q>();
                constexpr int offset = cx + cy * static_cast<int>(NX) + cz * static_cast<int>(STRIDE);

                const natural_t src = idx - static_cast<natural_t>(offset);

                const real_t cu = static_cast<real_t>(cx) * moments[midx(src, UX)] +
                                  static_cast<real_t>(cy) * moments[midx(src, UY)] +
                                  static_cast<real_t>(cz) * moments[midx(src, UZ)];

                const real_t mh = moments[midx(src, MXX)] * VelocitySet::hxx<Q>() +
                                  moments[midx(src, MYY)] * VelocitySet::hyy<Q>() +
                                  moments[midx(src, MZZ)] * VelocitySet::hzz<Q>() +
                                  moments[midx(src, MXY)] * VelocitySet::hxy<Q>() +
                                  moments[midx(src, MXZ)] * VelocitySet::hxz<Q>() +
                                  moments[midx(src, MYZ)] * VelocitySet::hyz<Q>();

                const real_t wrho = VelocitySet::w<Q>() * moments[midx(src, RHO)];
                const real_t fi = __fmaf_rn(wrho, cu + mh, wrho);

                rho += fi;
                ux += fi * static_cast<real_t>(cx);
                uy += fi * static_cast<real_t>(cy);
                uz += fi * static_cast<real_t>(cz);
                mxx += fi * static_cast<real_t>(cx * cx);
                myy += fi * static_cast<real_t>(cy * cy);
                mzz += fi * static_cast<real_t>(cz * cz);
                mxy += fi * static_cast<real_t>(cx * cy);
                mxz += fi * static_cast<real_t>(cx * cz);
                myz += fi * static_cast<real_t>(cy * cz);
            });

        const real_t invRho = static_cast<real_t>(1) / rho;

        ux *= invRho;
        uy *= invRho;
        uz *= invRho;
        mxx = __fmaf_rn(mxx, invRho, -VelocitySet::cs2());
        myy = __fmaf_rn(myy, invRho, -VelocitySet::cs2());
        mzz = __fmaf_rn(mzz, invRho, -VelocitySet::cs2());
        mxy *= invRho;
        mxz *= invRho;
        myz *= invRho;
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

    mxx = __fmaf_rn(OMEGA_D2 * ux, ux, T_OMEGA * mxx);
    myy = __fmaf_rn(OMEGA_D2 * uy, uy, T_OMEGA * myy);
    mzz = __fmaf_rn(OMEGA_D2 * uz, uz, T_OMEGA * mzz);
    mxy = __fmaf_rn(OMEGA * ux, uy, T_OMEGA * mxy);
    mxz = __fmaf_rn(OMEGA * ux, uz, T_OMEGA * mxz);
    myz = __fmaf_rn(OMEGA * uy, uz, T_OMEGA * myz);

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
