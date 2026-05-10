// const real_t ubx = ((nodeType & FRONT) == FRONT) ? CHAR_VELOCITY : static_cast<real_t>(0);
// const real_t uby = static_cast<real_t>(0);
// const real_t ubz = static_cast<real_t>(0);
// const real_t ub2 = ubx * ubx + uby * uby + ubz * ubz;

// real_t candA[7][7];
// real_t candB[7];

// for (natural_t r = 0; r < 7; ++r)
// {
//     candB[r] = static_cast<real_t>(0);
//     for (natural_t c = 0; c < 7; ++c)
//     {
//         candA[r][c] = static_cast<real_t>(0);
//     }
//     candA[r][r] = static_cast<real_t>(1);
// }

// constexpr_for<static_cast<natural_t>(0), static_cast<natural_t>(Q)>(
//     [&](const auto qConst) noexcept
//     {
//         constexpr natural_t q = qConst();
//         constexpr int cxi = CX[q];
//         constexpr int cyi = CY[q];
//         constexpr int czi = CZ[q];
//         constexpr real_t cx = static_cast<real_t>(cxi);
//         constexpr real_t cy = static_cast<real_t>(cyi);
//         constexpr real_t cz = static_cast<real_t>(czi);
//         constexpr real_t w = W[q];

//         const bool missing =
//             (((nodeType & WEST) == WEST) && (cxi > 0)) ||
//             (((nodeType & EAST) == EAST) && (cxi < 0)) ||
//             (((nodeType & SOUTH) == SOUTH) && (cyi > 0)) ||
//             (((nodeType & NORTH) == NORTH) && (cyi < 0)) ||
//             (((nodeType & BACK) == BACK) && (czi > 0)) ||
//             (((nodeType & FRONT) == FRONT) && (czi < 0));

//         const real_t hxx = cx * cx - CS2;
//         const real_t hyy = cy * cy - CS2;
//         const real_t hzz = cz * cz - CS2;
//         const real_t hxy = cx * cy;
//         const real_t hxz = cx * cz;
//         const real_t hyz = cy * cz;
//         const real_t H[7] = {
//             static_cast<real_t>(1),
//             hxx,
//             hyy,
//             hzz,
//             hxy,
//             hxz,
//             hyz};

//         if (missing)
//         {
//             const real_t uc = ubx * cx + uby * cy + ubz * cz;
//             real_t coeff[7];
//             coeff[0] = w * (static_cast<real_t>(1) + AS2 * uc);
//             coeff[1] = w * static_cast<real_t>(0.5) * AS4 * hxx;
//             coeff[2] = w * static_cast<real_t>(0.5) * AS4 * hyy;
//             coeff[3] = w * static_cast<real_t>(0.5) * AS4 * hzz;
//             coeff[4] = w * AS4 * hxy;
//             coeff[5] = w * AS4 * hxz;
//             coeff[6] = w * AS4 * hyz;

//             const real_t feqCoeff0 =
//                 coeff[0] +
//                 coeff[1] * ubx * ubx +
//                 coeff[2] * uby * uby +
//                 coeff[3] * ubz * ubz +
//                 coeff[4] * ubx * uby +
//                 coeff[5] * ubx * ubz +
//                 coeff[6] * uby * ubz;

//             real_t postCoeff[7];
//             postCoeff[0] = T_OMEGA * coeff[0] + OMEGA * feqCoeff0;
//             for (natural_t c = 1; c < 7; ++c)
//             {
//                 postCoeff[c] = T_OMEGA * coeff[c];
//             }

//             for (natural_t r = 0; r < 7; ++r)
//             {
//                 for (natural_t c = 0; c < 7; ++c)
//                 {
//                     candA[r][c] -= H[r] * postCoeff[c];
//                 }
//             }
//         }
//         else
//         {
//             const int xs_i = static_cast<int>(x) - cxi;
//             const int ys_i = static_cast<int>(y) - cyi;
//             const int zs_i = static_cast<int>(z) - czi;

//             if (xs_i >= 0 && ys_i >= 0 && zs_i >= 0 &&
//                 xs_i < static_cast<int>(NX) &&
//                 ys_i < static_cast<int>(NY) &&
//                 zs_i < static_cast<int>(NZ))
//             {
//                 const natural_t xs = static_cast<natural_t>(xs_i);
//                 const natural_t ys = static_cast<natural_t>(ys_i);
//                 const natural_t zs = static_cast<natural_t>(zs_i);
//                 const natural_t txs = xs % BLOCK_NX;
//                 const natural_t tys = ys % BLOCK_NY;
//                 const natural_t tzs = zs % BLOCK_NZ;
//                 const natural_t bxs = xs / BLOCK_NX;
//                 const natural_t bys = ys / BLOCK_NY;
//                 const natural_t bzs = zs / BLOCK_NZ;
//                 const natural_t src = idx(txs, tys, tzs, bxs, bys, bzs);

//                 const real_t rho_s = moments[src + CELLS * RHO];
//                 const real_t ux_s = moments[src + CELLS * UX];
//                 const real_t uy_s = moments[src + CELLS * UY];
//                 const real_t uz_s = moments[src + CELLS * UZ];
//                 const real_t mxx_s = moments[src + CELLS * MXX];
//                 const real_t myy_s = moments[src + CELLS * MYY];
//                 const real_t mzz_s = moments[src + CELLS * MZZ];
//                 const real_t mxy_s = moments[src + CELLS * MXY];
//                 const real_t mxz_s = moments[src + CELLS * MXZ];
//                 const real_t myz_s = moments[src + CELLS * MYZ];

//                 const real_t cu = cx * ux_s + cy * uy_s + cz * uz_s;
//                 const real_t mh =
//                     mxx_s * hxx +
//                     myy_s * hyy +
//                     mzz_s * hzz +
//                     mxy_s * hxy +
//                     mxz_s * hxz +
//                     myz_s * hyz;
//                 const real_t fq = w * rho_s * (static_cast<real_t>(1) + cu + mh);

//                 for (natural_t r = 0; r < 7; ++r)
//                 {
//                     candB[r] += fq * H[r];
//                 }
//             }
//         }
//     });

// real_t A[7][8];
// for (natural_t r = 0; r < 7; ++r)
// {
//     for (natural_t c = 0; c < 8; ++c)
//     {
//         A[r][c] = static_cast<real_t>(0);
//     }
// }

// A[0][0] = -ub2;
// A[0][1] = static_cast<real_t>(1);
// A[0][2] = static_cast<real_t>(1);
// A[0][3] = static_cast<real_t>(1);
// A[0][7] = static_cast<real_t>(0);

// natural_t rowCount = 1;

// for (natural_t cand = 0; cand < 7 && rowCount < 7; ++cand)
// {
//     real_t rankMat[7][7];
//     for (natural_t r = 0; r < 7; ++r)
//     {
//         for (natural_t c = 0; c < 7; ++c)
//         {
//             rankMat[r][c] = static_cast<real_t>(0);
//         }
//     }

//     for (natural_t r = 0; r < rowCount; ++r)
//     {
//         for (natural_t c = 0; c < 7; ++c)
//         {
//             rankMat[r][c] = A[r][c];
//         }
//     }
//     for (natural_t c = 0; c < 7; ++c)
//     {
//         rankMat[rowCount][c] = candA[cand][c];
//     }

//     natural_t rank = 0;
//     for (natural_t col = 0; col < 7 && rank < rowCount + 1; ++col)
//     {
//         natural_t pivot = rank;
//         real_t maxVal = rankMat[pivot][col];
//         maxVal = (maxVal < static_cast<real_t>(0)) ? -maxVal : maxVal;

//         for (natural_t r = rank + 1; r < rowCount + 1; ++r)
//         {
//             real_t val = rankMat[r][col];
//             val = (val < static_cast<real_t>(0)) ? -val : val;
//             if (val > maxVal)
//             {
//                 maxVal = val;
//                 pivot = r;
//             }
//         }

//         if (maxVal > EPS)
//         {
//             if (pivot != rank)
//             {
//                 for (natural_t c = col; c < 7; ++c)
//                 {
//                     const real_t tmp = rankMat[rank][c];
//                     rankMat[rank][c] = rankMat[pivot][c];
//                     rankMat[pivot][c] = tmp;
//                 }
//             }

//             const real_t invPivot = static_cast<real_t>(1) / rankMat[rank][col];
//             for (natural_t r = rank + 1; r < rowCount + 1; ++r)
//             {
//                 const real_t factor = rankMat[r][col] * invPivot;
//                 for (natural_t c = col; c < 7; ++c)
//                 {
//                     rankMat[r][c] -= factor * rankMat[rank][c];
//                 }
//             }
//             ++rank;
//         }
//     }

//     if (rank > rowCount)
//     {
//         for (natural_t c = 0; c < 7; ++c)
//         {
//             A[rowCount][c] = candA[cand][c];
//         }
//         A[rowCount][7] = candB[cand];
//         ++rowCount;
//     }
// }

// bool valid = (rowCount == 7);

// for (natural_t col = 0; col < 7 && valid; ++col)
// {
//     natural_t pivot = col;
//     real_t maxVal = A[pivot][col];
//     maxVal = (maxVal < static_cast<real_t>(0)) ? -maxVal : maxVal;

//     for (natural_t r = col + 1; r < 7; ++r)
//     {
//         real_t val = A[r][col];
//         val = (val < static_cast<real_t>(0)) ? -val : val;
//         if (val > maxVal)
//         {
//             maxVal = val;
//             pivot = r;
//         }
//     }

//     if (maxVal <= EPS)
//     {
//         valid = false;
//     }
//     else
//     {
//         if (pivot != col)
//         {
//             for (natural_t c = col; c < 8; ++c)
//             {
//                 const real_t tmp = A[col][c];
//                 A[col][c] = A[pivot][c];
//                 A[pivot][c] = tmp;
//             }
//         }

//         const real_t invPivot = static_cast<real_t>(1) / A[col][col];
//         for (natural_t r = col + 1; r < 7; ++r)
//         {
//             const real_t factor = A[r][col] * invPivot;
//             for (natural_t c = col; c < 8; ++c)
//             {
//                 A[r][c] -= factor * A[col][c];
//             }
//         }
//     }
// }

// real_t X[7];
// for (natural_t r = 0; r < 7; ++r)
// {
//     X[r] = static_cast<real_t>(0);
// }

// if (valid)
// {
//     for (int r = 6; r >= 0; --r)
//     {
//         real_t rhs = A[r][7];
//         for (natural_t c = static_cast<natural_t>(r + 1); c < 7; ++c)
//         {
//             rhs -= A[r][c] * X[c];
//         }

//         X[r] = rhs / A[r][r];
//     }

//     valid = (X[0] > EPS) && (X[0] == X[0]);
// }

// if (valid)
// {
//     rho = X[0];
//     ux = ubx;
//     uy = uby;
//     uz = ubz;

//     const real_t invRho = static_cast<real_t>(1) / rho;
//     mxx = X[1] * invRho;
//     myy = X[2] * invRho;
//     mzz = X[3] * invRho;
//     mxy = X[4] * invRho;
//     mxz = X[5] * invRho;
//     myz = X[6] * invRho;
// }
// else
// {
//     rho = moments[idx_ + CELLS * RHO];
//     ux = ubx;
//     uy = uby;
//     uz = ubz;
//     mxx = ux * ux;
//     myy = uy * uy;
//     mzz = uz * uz;
//     mxy = ux * uy;
//     mxz = ux * uz;
//     myz = uy * uz;
// }
