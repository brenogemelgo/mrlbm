#pragma once

#include "bitmasks.cuh"
#include "constants.cuh"
#include "deviceFunctions.cuh"

#include <cuda_runtime.h>

constexpr natural_t IRBC_UNKNOWNS = 7;
constexpr natural_t IRBC_TABLE_STRIDE = IRBC_UNKNOWNS * IRBC_UNKNOWNS;
constexpr natural_t IRBC_TABLE_SIZE = 64 * IRBC_TABLE_STRIDE;

__device__ __constant__ real_t IRBC_INVERSE[IRBC_TABLE_SIZE];

template <natural_t q>
__device__ [[nodiscard]] static __forceinline__ constexpr int cxValue() noexcept
{
    if constexpr (q == 1 || q == 7 || q == 9 || q == 13 || q == 15 || q == 19 || q == 21 || q == 23 || q == 26)
    {
        return 1;
    }
    else if constexpr (q == 2 || q == 8 || q == 10 || q == 14 || q == 16 || q == 20 || q == 22 || q == 24 || q == 25)
    {
        return -1;
    }
    else
    {
        return 0;
    }
}

template <natural_t q>
__device__ [[nodiscard]] static __forceinline__ constexpr int cyValue() noexcept
{
    if constexpr (q == 3 || q == 7 || q == 11 || q == 14 || q == 17 || q == 19 || q == 21 || q == 24 || q == 25)
    {
        return 1;
    }
    else if constexpr (q == 4 || q == 8 || q == 12 || q == 13 || q == 18 || q == 20 || q == 22 || q == 23 || q == 26)
    {
        return -1;
    }
    else
    {
        return 0;
    }
}

template <natural_t q>
__device__ [[nodiscard]] static __forceinline__ constexpr int czValue() noexcept
{
    if constexpr (q == 5 || q == 9 || q == 11 || q == 16 || q == 18 || q == 19 || q == 22 || q == 23 || q == 25)
    {
        return 1;
    }
    else if constexpr (q == 6 || q == 10 || q == 12 || q == 15 || q == 17 || q == 20 || q == 21 || q == 24 || q == 26)
    {
        return -1;
    }
    else
    {
        return 0;
    }
}

template <natural_t q>
__device__ [[nodiscard]] static __forceinline__ constexpr real_t wValue() noexcept
{
    if constexpr (q == 0)
    {
        return W1;
    }
    else if constexpr (q <= 6)
    {
        return W2;
    }
    else if constexpr (q <= 18)
    {
        return W3;
    }
    else
    {
        return W4;
    }
}

template <natural_t q>
__device__ [[nodiscard]] static __forceinline__ constexpr real_t hxxValue() noexcept
{
    return static_cast<real_t>(cxValue<q>() * cxValue<q>()) - CS2;
}

template <natural_t q>
__device__ [[nodiscard]] static __forceinline__ constexpr real_t hyyValue() noexcept
{
    return static_cast<real_t>(cyValue<q>() * cyValue<q>()) - CS2;
}

template <natural_t q>
__device__ [[nodiscard]] static __forceinline__ constexpr real_t hzzValue() noexcept
{
    return static_cast<real_t>(czValue<q>() * czValue<q>()) - CS2;
}

template <natural_t q>
__device__ [[nodiscard]] static __forceinline__ constexpr real_t hxyValue() noexcept
{
    return static_cast<real_t>(cxValue<q>() * cyValue<q>());
}

template <natural_t q>
__device__ [[nodiscard]] static __forceinline__ constexpr real_t hxzValue() noexcept
{
    return static_cast<real_t>(cxValue<q>() * czValue<q>());
}

template <natural_t q>
__device__ [[nodiscard]] static __forceinline__ constexpr real_t hyzValue() noexcept
{
    return static_cast<real_t>(cyValue<q>() * czValue<q>());
}

template <natural_t q>
__device__ static __forceinline__ void hermiteBasis(real_t (&h)[6]) noexcept
{
    h[0] = hxxValue<q>();
    h[1] = hyyValue<q>();
    h[2] = hzzValue<q>();
    h[3] = hxyValue<q>();
    h[4] = hxzValue<q>();
    h[5] = hyzValue<q>();
}

template <natural_t q>
__device__ [[nodiscard]] static __forceinline__ bool isMissingDirection(const unsigned int nodeType) noexcept
{
    return (((nodeType & WEST) == WEST) && (cxValue<q>() > 0)) ||
           (((nodeType & EAST) == EAST) && (cxValue<q>() < 0)) ||
           (((nodeType & SOUTH) == SOUTH) && (cyValue<q>() > 0)) ||
           (((nodeType & NORTH) == NORTH) && (cyValue<q>() < 0)) ||
           (((nodeType & BACK) == BACK) && (czValue<q>() > 0)) ||
           (((nodeType & FRONT) == FRONT) && (czValue<q>() < 0));
}

template <unsigned int nodeTypeValue, natural_t q>
__device__ [[nodiscard]] static __forceinline__ constexpr bool isMissingDirectionConst() noexcept
{
    return (((nodeTypeValue & WEST) == WEST) && (cxValue<q>() > 0)) ||
           (((nodeTypeValue & EAST) == EAST) && (cxValue<q>() < 0)) ||
           (((nodeTypeValue & SOUTH) == SOUTH) && (cyValue<q>() > 0)) ||
           (((nodeTypeValue & NORTH) == NORTH) && (cyValue<q>() < 0)) ||
           (((nodeTypeValue & BACK) == BACK) && (czValue<q>() > 0)) ||
           (((nodeTypeValue & FRONT) == FRONT) && (czValue<q>() < 0));
}

template <natural_t q>
__device__ [[nodiscard]] static __forceinline__ real_t reconstructStreamedPopulation(
    const real_t *__restrict__ moments,
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    const natural_t xs = static_cast<natural_t>(static_cast<int>(x) - cxValue<q>());
    const natural_t ys = static_cast<natural_t>(static_cast<int>(y) - cyValue<q>());
    const natural_t zs = static_cast<natural_t>(static_cast<int>(z) - czValue<q>());
    const natural_t src = global3(
        xs,
        ys,
        zs);

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

    const real_t cu = static_cast<real_t>(cxValue<q>()) * ux_s +
                      static_cast<real_t>(cyValue<q>()) * uy_s +
                      static_cast<real_t>(czValue<q>()) * uz_s;

    const real_t mh = mxx_s * hxxValue<q>() +
                      myy_s * hyyValue<q>() +
                      mzz_s * hzzValue<q>() +
                      mxy_s * hxyValue<q>() +
                      mxz_s * hxzValue<q>() +
                      myz_s * hyzValue<q>();

    return wValue<q>() * rho_s * (static_cast<real_t>(1) + cu + mh);
}

__device__ static __forceinline__ void boundaryVelocity(
    const unsigned int nodeType,
    real_t &ubx,
    real_t &uby,
    real_t &ubz) noexcept
{
    if ((nodeType & FRONT) == FRONT)
    {
        ubx = CHAR_VELOCITY;
        uby = static_cast<real_t>(0);
        ubz = static_cast<real_t>(0);
    }
    else
    {
        ubx = static_cast<real_t>(0);
        uby = static_cast<real_t>(0);
        ubz = static_cast<real_t>(0);
    }
}

template <unsigned int nodeTypeValue>
__device__ static __forceinline__ void boundaryVelocityConst(
    real_t &ubx,
    real_t &uby,
    real_t &ubz) noexcept
{
    if constexpr ((nodeTypeValue & FRONT) == FRONT)
    {
        ubx = CHAR_VELOCITY;
        uby = static_cast<real_t>(0);
        ubz = static_cast<real_t>(0);
    }
    else
    {
        ubx = static_cast<real_t>(0);
        uby = static_cast<real_t>(0);
        ubz = static_cast<real_t>(0);
    }
}

template <unsigned int nodeTypeValue>
__device__ static __forceinline__ void applyIRBCBoundaryTyped(
    const real_t *__restrict__ moments,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    real_t &rho,
    real_t &ux,
    real_t &uy,
    real_t &uz,
    real_t &mxx,
    real_t &myy,
    real_t &mzz,
    real_t &mxy,
    real_t &mxz,
    real_t &myz) noexcept
{
    constexpr natural_t tableOffset = nodeTypeValue * IRBC_TABLE_STRIDE;

    real_t ubx;
    real_t uby;
    real_t ubz;
    boundaryVelocityConst<nodeTypeValue>(ubx, uby, ubz);

    real_t rhs[IRBC_UNKNOWNS] = {
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0)};

    constexpr_for<static_cast<natural_t>(0), static_cast<natural_t>(Q)>(
        [&](const auto qConst) noexcept
        {
            constexpr natural_t q = qConst();

            if constexpr (!isMissingDirectionConst<nodeTypeValue, q>())
            {
                const real_t f = reconstructStreamedPopulation<q>(moments, x, y, z);
                rhs[0] += f;
                rhs[1] += f * (hxxValue<q>() - hzzValue<q>());
                rhs[2] += f * (hyyValue<q>() - hzzValue<q>());
                rhs[3] += f * hxyValue<q>();
                rhs[4] += f * hxzValue<q>();
                rhs[5] += f * hyzValue<q>();
            }
        });

    rhs[6] = static_cast<real_t>(0);

    real_t solved[IRBC_UNKNOWNS] = {
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0)};

#pragma unroll
    for (natural_t row = 0; row < IRBC_UNKNOWNS; ++row)
    {
#pragma unroll
        for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
        {
            solved[row] += IRBC_INVERSE[tableOffset + row * IRBC_UNKNOWNS + col] * rhs[col];
        }
    }

    rho = solved[0];
    const real_t invRho = static_cast<real_t>(1) / rho;

    ux = ubx;
    uy = uby;
    uz = ubz;
    mxx = solved[1] * invRho;
    myy = solved[2] * invRho;
    mzz = solved[3] * invRho;
    mxy = solved[4] * invRho;
    mxz = solved[5] * invRho;
    myz = solved[6] * invRho;
}

__device__ static __forceinline__ void applyIRBCBoundary(
    const real_t *__restrict__ moments,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    const unsigned int nodeType,
    real_t &rho,
    real_t &ux,
    real_t &uy,
    real_t &uz,
    real_t &mxx,
    real_t &myy,
    real_t &mzz,
    real_t &mxy,
    real_t &mxz,
    real_t &myz) noexcept
{
    const natural_t tableOffset = static_cast<natural_t>(nodeType) * IRBC_TABLE_STRIDE;

    real_t ubx;
    real_t uby;
    real_t ubz;
    boundaryVelocity(nodeType, ubx, uby, ubz);

    real_t rhs[IRBC_UNKNOWNS] = {
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0)};

    constexpr_for<static_cast<natural_t>(0), static_cast<natural_t>(Q)>(
        [&](const auto qConst) noexcept
        {
            constexpr natural_t q = qConst();

            if (!isMissingDirection<q>(nodeType))
            {
                const real_t f = reconstructStreamedPopulation<q>(moments, x, y, z);
                rhs[0] += f;
                rhs[1] += f * (hxxValue<q>() - hzzValue<q>());
                rhs[2] += f * (hyyValue<q>() - hzzValue<q>());
                rhs[3] += f * hxyValue<q>();
                rhs[4] += f * hxzValue<q>();
                rhs[5] += f * hyzValue<q>();
            }
        });

    rhs[6] = static_cast<real_t>(0);

    real_t solved[IRBC_UNKNOWNS] = {
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0)};

#pragma unroll
    for (natural_t row = 0; row < IRBC_UNKNOWNS; ++row)
    {
#pragma unroll
        for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
        {
            solved[row] += IRBC_INVERSE[tableOffset + row * IRBC_UNKNOWNS + col] * rhs[col];
        }
    }

    rho = solved[0];
    const real_t invRho = static_cast<real_t>(1) / rho;

    ux = ubx;
    uy = uby;
    uz = ubz;
    mxx = solved[1] * invRho;
    myy = solved[2] * invRho;
    mzz = solved[3] * invRho;
    mxy = solved[4] * invRho;
    mxz = solved[5] * invRho;
    myz = solved[6] * invRho;
}

__device__ static __forceinline__ void dispatchIRBCBoundary(
    const real_t *__restrict__ moments,
    const natural_t x,
    const natural_t y,
    const natural_t z,
    const unsigned int nodeType,
    real_t &rho,
    real_t &ux,
    real_t &uy,
    real_t &uz,
    real_t &mxx,
    real_t &myy,
    real_t &mzz,
    real_t &mxy,
    real_t &mxz,
    real_t &myz) noexcept
{
    switch (nodeType)
    {
    case WEST_FACE:
        applyIRBCBoundaryTyped<WEST_FACE>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case EAST_FACE:
        applyIRBCBoundaryTyped<EAST_FACE>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case SOUTH_FACE:
        applyIRBCBoundaryTyped<SOUTH_FACE>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case NORTH_FACE:
        applyIRBCBoundaryTyped<NORTH_FACE>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case BACK_FACE:
        applyIRBCBoundaryTyped<BACK_FACE>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case FRONT_FACE:
        applyIRBCBoundaryTyped<FRONT_FACE>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case NORTH_WEST:
        applyIRBCBoundaryTyped<NORTH_WEST>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case NORTH_EAST:
        applyIRBCBoundaryTyped<NORTH_EAST>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case NORTH_FRONT:
        applyIRBCBoundaryTyped<NORTH_FRONT>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case NORTH_BACK:
        applyIRBCBoundaryTyped<NORTH_BACK>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case SOUTH_WEST:
        applyIRBCBoundaryTyped<SOUTH_WEST>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case SOUTH_EAST:
        applyIRBCBoundaryTyped<SOUTH_EAST>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case SOUTH_FRONT:
        applyIRBCBoundaryTyped<SOUTH_FRONT>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case SOUTH_BACK:
        applyIRBCBoundaryTyped<SOUTH_BACK>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case WEST_FRONT:
        applyIRBCBoundaryTyped<WEST_FRONT>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case WEST_BACK:
        applyIRBCBoundaryTyped<WEST_BACK>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case EAST_FRONT:
        applyIRBCBoundaryTyped<EAST_FRONT>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case EAST_BACK:
        applyIRBCBoundaryTyped<EAST_BACK>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case NORTH_WEST_FRONT:
        applyIRBCBoundaryTyped<NORTH_WEST_FRONT>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case NORTH_WEST_BACK:
        applyIRBCBoundaryTyped<NORTH_WEST_BACK>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case NORTH_EAST_FRONT:
        applyIRBCBoundaryTyped<NORTH_EAST_FRONT>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case NORTH_EAST_BACK:
        applyIRBCBoundaryTyped<NORTH_EAST_BACK>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case SOUTH_WEST_FRONT:
        applyIRBCBoundaryTyped<SOUTH_WEST_FRONT>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case SOUTH_WEST_BACK:
        applyIRBCBoundaryTyped<SOUTH_WEST_BACK>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case SOUTH_EAST_FRONT:
        applyIRBCBoundaryTyped<SOUTH_EAST_FRONT>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    case SOUTH_EAST_BACK:
        applyIRBCBoundaryTyped<SOUTH_EAST_BACK>(moments, x, y, z, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    default:
        applyIRBCBoundary(moments, x, y, z, nodeType, rho, ux, uy, uz, mxx, myy, mzz, mxy, mxz, myz);
        return;
    }
}

__host__ [[nodiscard]] static inline bool isValidBoundaryType(const unsigned int nodeType) noexcept
{
    if (nodeType == BULK)
    {
        return false;
    }

    return !(((nodeType & WEST) != 0u && (nodeType & EAST) != 0u) ||
             ((nodeType & SOUTH) != 0u && (nodeType & NORTH) != 0u) ||
             ((nodeType & BACK) != 0u && (nodeType & FRONT) != 0u));
}

__host__ [[nodiscard]] static inline bool isMissingHost(const unsigned int nodeType, const natural_t q) noexcept
{
    return (((nodeType & WEST) == WEST) && (CX[q] > 0)) ||
           (((nodeType & EAST) == EAST) && (CX[q] < 0)) ||
           (((nodeType & SOUTH) == SOUTH) && (CY[q] > 0)) ||
           (((nodeType & NORTH) == NORTH) && (CY[q] < 0)) ||
           (((nodeType & BACK) == BACK) && (CZ[q] > 0)) ||
           (((nodeType & FRONT) == FRONT) && (CZ[q] < 0));
}

__host__ static inline void boundaryVelocityHost(
    const unsigned int nodeType,
    real_t &ubx,
    real_t &uby,
    real_t &ubz) noexcept
{
    if ((nodeType & FRONT) == FRONT)
    {
        ubx = CHAR_VELOCITY;
        uby = static_cast<real_t>(0);
        ubz = static_cast<real_t>(0);
    }
    else
    {
        ubx = static_cast<real_t>(0);
        uby = static_cast<real_t>(0);
        ubz = static_cast<real_t>(0);
    }
}

__host__ static inline void invertIRBCMatrix(
    real_t (&a)[IRBC_UNKNOWNS][IRBC_UNKNOWNS],
    real_t (&inv)[IRBC_UNKNOWNS][IRBC_UNKNOWNS]) noexcept
{
    for (natural_t row = 0; row < IRBC_UNKNOWNS; ++row)
    {
        for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
        {
            inv[row][col] = row == col ? static_cast<real_t>(1) : static_cast<real_t>(0);
        }
    }

    for (natural_t pivot = 0; pivot < IRBC_UNKNOWNS; ++pivot)
    {
        natural_t pivotRow = pivot;
        real_t pivotAbs = a[pivot][pivot] < static_cast<real_t>(0) ? -a[pivot][pivot] : a[pivot][pivot];

        for (natural_t row = pivot + 1; row < IRBC_UNKNOWNS; ++row)
        {
            const real_t valueAbs = a[row][pivot] < static_cast<real_t>(0) ? -a[row][pivot] : a[row][pivot];
            if (valueAbs > pivotAbs)
            {
                pivotAbs = valueAbs;
                pivotRow = row;
            }
        }

        if (pivotRow != pivot)
        {
            for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
            {
                const real_t tmpA = a[pivot][col];
                a[pivot][col] = a[pivotRow][col];
                a[pivotRow][col] = tmpA;

                const real_t tmpInv = inv[pivot][col];
                inv[pivot][col] = inv[pivotRow][col];
                inv[pivotRow][col] = tmpInv;
            }
        }

        const real_t pivotValue = a[pivot][pivot];
        const real_t invPivot = static_cast<real_t>(1) / pivotValue;

        for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
        {
            a[pivot][col] *= invPivot;
            inv[pivot][col] *= invPivot;
        }

        for (natural_t row = 0; row < IRBC_UNKNOWNS; ++row)
        {
            if (row == pivot)
            {
                continue;
            }

            const real_t factor = a[row][pivot];
            for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
            {
                a[row][col] -= factor * a[pivot][col];
                inv[row][col] -= factor * inv[pivot][col];
            }
        }
    }
}

__host__ static inline void assembleIRBCInverse(
    const unsigned int nodeType,
    real_t (&invOut)[IRBC_UNKNOWNS][IRBC_UNKNOWNS]) noexcept
{
    real_t ubx;
    real_t uby;
    real_t ubz;
    boundaryVelocityHost(nodeType, ubx, uby, ubz);
    const real_t ub2 = ubx * ubx + uby * uby + ubz * ubz;

    real_t density[IRBC_UNKNOWNS] = {
        static_cast<real_t>(1),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0),
        static_cast<real_t>(0)};

    real_t momentRows[6][IRBC_UNKNOWNS] = {};
    momentRows[0][1] = static_cast<real_t>(1);
    momentRows[1][2] = static_cast<real_t>(1);
    momentRows[2][3] = static_cast<real_t>(1);
    momentRows[3][4] = static_cast<real_t>(1);
    momentRows[4][5] = static_cast<real_t>(1);
    momentRows[5][6] = static_cast<real_t>(1);

    for (natural_t q = 0; q < Q; ++q)
    {
        if (!isMissingHost(nodeType, q))
        {
            continue;
        }

        const real_t cx = static_cast<real_t>(CX[q]);
        const real_t cy = static_cast<real_t>(CY[q]);
        const real_t cz = static_cast<real_t>(CZ[q]);
        const real_t h[6] = {
            cx * cx - CS2,
            cy * cy - CS2,
            cz * cz - CS2,
            cx * cy,
            cx * cz,
            cy * cz};

        const real_t cu = ubx * cx + uby * cy + ubz * cz;
        const real_t meqH =
            static_cast<real_t>(0.5) * AS4 * (ubx * ubx * h[0] + uby * uby * h[1] + ubz * ubz * h[2]) +
            AS4 * (ubx * uby * h[3] + ubx * ubz * h[4] + uby * ubz * h[5]);
        const real_t coeff[IRBC_UNKNOWNS] = {
            W[q] * (static_cast<real_t>(1) + AS2 * cu + OMEGA * meqH),
            W[q] * T_OMEGA * static_cast<real_t>(0.5) * AS4 * h[0],
            W[q] * T_OMEGA * static_cast<real_t>(0.5) * AS4 * h[1],
            W[q] * T_OMEGA * static_cast<real_t>(0.5) * AS4 * h[2],
            W[q] * T_OMEGA * AS4 * h[3],
            W[q] * T_OMEGA * AS4 * h[4],
            W[q] * T_OMEGA * AS4 * h[5]};

        for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
        {
            density[col] -= coeff[col];
        }

        for (natural_t row = 0; row < 6; ++row)
        {
            for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
            {
                momentRows[row][col] -= h[row] * coeff[col];
            }
        }
    }

    real_t matrix[IRBC_UNKNOWNS][IRBC_UNKNOWNS] = {};

    for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
    {
        matrix[0][col] = density[col];
        matrix[1][col] = momentRows[0][col] - momentRows[2][col];
        matrix[2][col] = momentRows[1][col] - momentRows[2][col];
        matrix[3][col] = momentRows[3][col];
        matrix[4][col] = momentRows[4][col];
        matrix[5][col] = momentRows[5][col];
        matrix[6][col] = static_cast<real_t>(0);
    }

    matrix[6][0] = -ub2;
    matrix[6][1] = static_cast<real_t>(1);
    matrix[6][2] = static_cast<real_t>(1);
    matrix[6][3] = static_cast<real_t>(1);

    invertIRBCMatrix(matrix, invOut);
}

__host__ [[nodiscard]] static inline cudaError_t initIRBCBoundaryTables() noexcept
{
    real_t hostTable[IRBC_TABLE_SIZE] = {};

    for (unsigned int nodeType = 0; nodeType < 64; ++nodeType)
    {
        if (!isValidBoundaryType(nodeType))
        {
            continue;
        }

        real_t inv[IRBC_UNKNOWNS][IRBC_UNKNOWNS] = {};
        assembleIRBCInverse(nodeType, inv);

        const natural_t tableOffset = static_cast<natural_t>(nodeType) * IRBC_TABLE_STRIDE;
        for (natural_t row = 0; row < IRBC_UNKNOWNS; ++row)
        {
            for (natural_t col = 0; col < IRBC_UNKNOWNS; ++col)
            {
                hostTable[tableOffset + row * IRBC_UNKNOWNS + col] = inv[row][col];
            }
        }
    }

    return cudaMemcpyToSymbol(IRBC_INVERSE, hostTable, sizeof(hostTable));
}
