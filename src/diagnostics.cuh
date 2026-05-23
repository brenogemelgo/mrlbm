#pragma once

#include "bitmasks.cuh"
#include "deviceFunctions.cuh"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

constexpr natural_t DIAGNOSTIC_THREADS = 256;
constexpr natural_t DIAGNOSTIC_BLOCKS = 256;
constexpr natural_t DIAGNOSTIC_FIELDS = 5;
constexpr natural_t DIAGNOSTIC_STATS_FIELDS = 6;

constexpr natural_t DIAGNOSTIC_SUM_UX2 = 0;
constexpr natural_t DIAGNOSTIC_SUM_UY2 = 1;
constexpr natural_t DIAGNOSTIC_SUM_UZ2 = 2;
constexpr natural_t DIAGNOSTIC_SUM_TKE = 3;
constexpr natural_t DIAGNOSTIC_FLUID_COUNT = 4;

constexpr natural_t DIAGNOSTIC_MEAN_UX = 0;
constexpr natural_t DIAGNOSTIC_MEAN_UY = 1;
constexpr natural_t DIAGNOSTIC_MEAN_UZ = 2;
constexpr natural_t DIAGNOSTIC_MEAN_UX2 = 3;
constexpr natural_t DIAGNOSTIC_MEAN_UY2 = 4;
constexpr natural_t DIAGNOSTIC_MEAN_UZ2 = 5;

struct KineticEnergySample
{
    double keTotal = 0.0;
    double keMean = 0.0;
    double tke = 0.0;
    natural_t fluidCells = 0;
    natural_t statisticsSamples = 0;
};

struct KineticEnergyDiagnostics
{
    double *deviceBlockSums = nullptr;
    real_t *deviceStatistics = nullptr;
    std::vector<double> hostBlockSums;
    natural_t statisticsSamples = 0;
    std::ofstream keTotalOutput;
    std::ofstream keMeanOutput;
    std::ofstream tkeOutput;
};

__device__ [[nodiscard]] static __forceinline__ natural_t diagnosticStatIndex(
    const natural_t idx,
    const natural_t statistic) noexcept
{
    return idx + CELLS * statistic;
}

__global__ void updateKineticEnergyStatisticsKernel(
    const real_t *__restrict__ moments,
    real_t *__restrict__ statistics,
    const natural_t sampleCount)
{
    const natural_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const natural_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const natural_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= NX || y >= NY || z >= NZ || boundaryMask(x, y, z) != BULK)
    {
        return;
    }

    const natural_t idx = global3(x, y, z);
    const real_t invSampleCount = static_cast<real_t>(1) / static_cast<real_t>(sampleCount);

    const real_t ux = loadMoment(moments, idx, UX);
    const real_t uy = loadMoment(moments, idx, UY);
    const real_t uz = loadMoment(moments, idx, UZ);

    real_t meanUx = statistics[diagnosticStatIndex(idx, DIAGNOSTIC_MEAN_UX)];
    real_t meanUy = statistics[diagnosticStatIndex(idx, DIAGNOSTIC_MEAN_UY)];
    real_t meanUz = statistics[diagnosticStatIndex(idx, DIAGNOSTIC_MEAN_UZ)];
    real_t meanUx2 = statistics[diagnosticStatIndex(idx, DIAGNOSTIC_MEAN_UX2)];
    real_t meanUy2 = statistics[diagnosticStatIndex(idx, DIAGNOSTIC_MEAN_UY2)];
    real_t meanUz2 = statistics[diagnosticStatIndex(idx, DIAGNOSTIC_MEAN_UZ2)];

    meanUx = __fmaf_rn(ux - meanUx, invSampleCount, meanUx);
    meanUy = __fmaf_rn(uy - meanUy, invSampleCount, meanUy);
    meanUz = __fmaf_rn(uz - meanUz, invSampleCount, meanUz);
    meanUx2 = __fmaf_rn(ux * ux - meanUx2, invSampleCount, meanUx2);
    meanUy2 = __fmaf_rn(uy * uy - meanUy2, invSampleCount, meanUy2);
    meanUz2 = __fmaf_rn(uz * uz - meanUz2, invSampleCount, meanUz2);

    statistics[diagnosticStatIndex(idx, DIAGNOSTIC_MEAN_UX)] = meanUx;
    statistics[diagnosticStatIndex(idx, DIAGNOSTIC_MEAN_UY)] = meanUy;
    statistics[diagnosticStatIndex(idx, DIAGNOSTIC_MEAN_UZ)] = meanUz;
    statistics[diagnosticStatIndex(idx, DIAGNOSTIC_MEAN_UX2)] = meanUx2;
    statistics[diagnosticStatIndex(idx, DIAGNOSTIC_MEAN_UY2)] = meanUy2;
    statistics[diagnosticStatIndex(idx, DIAGNOSTIC_MEAN_UZ2)] = meanUz2;
}

__global__ void kineticEnergyPartialSums(
    const real_t *__restrict__ moments,
    const real_t *__restrict__ statistics,
    double *__restrict__ blockSums)
{
    __shared__ double shared[DIAGNOSTIC_FIELDS][DIAGNOSTIC_THREADS];

    double local[DIAGNOSTIC_FIELDS] = {};

    const natural_t thread = threadIdx.x;
    const natural_t stride = blockDim.x * gridDim.x;

    for (natural_t idx = blockIdx.x * blockDim.x + thread; idx < CELLS; idx += stride)
    {
        const natural_t z = idx / STRIDE;
        const natural_t xy = idx - z * STRIDE;
        const natural_t y = xy / NX;
        const natural_t x = xy - y * NX;

        if (boundaryMask(x, y, z) != BULK)
        {
            continue;
        }

        const double ux = static_cast<double>(loadMoment(moments, idx, UX));
        const double uy = static_cast<double>(loadMoment(moments, idx, UY));
        const double uz = static_cast<double>(loadMoment(moments, idx, UZ));

        const double meanUx = static_cast<double>(statistics[diagnosticStatIndex(idx, DIAGNOSTIC_MEAN_UX)]);
        const double meanUy = static_cast<double>(statistics[diagnosticStatIndex(idx, DIAGNOSTIC_MEAN_UY)]);
        const double meanUz = static_cast<double>(statistics[diagnosticStatIndex(idx, DIAGNOSTIC_MEAN_UZ)]);
        const double meanUx2 = static_cast<double>(statistics[diagnosticStatIndex(idx, DIAGNOSTIC_MEAN_UX2)]);
        const double meanUy2 = static_cast<double>(statistics[diagnosticStatIndex(idx, DIAGNOSTIC_MEAN_UY2)]);
        const double meanUz2 = static_cast<double>(statistics[diagnosticStatIndex(idx, DIAGNOSTIC_MEAN_UZ2)]);

        const double rawVarUx = meanUx2 - meanUx * meanUx;
        const double rawVarUy = meanUy2 - meanUy * meanUy;
        const double rawVarUz = meanUz2 - meanUz * meanUz;
        const double varUx = rawVarUx > 0.0 ? rawVarUx : 0.0;
        const double varUy = rawVarUy > 0.0 ? rawVarUy : 0.0;
        const double varUz = rawVarUz > 0.0 ? rawVarUz : 0.0;

        local[DIAGNOSTIC_SUM_UX2] += ux * ux;
        local[DIAGNOSTIC_SUM_UY2] += uy * uy;
        local[DIAGNOSTIC_SUM_UZ2] += uz * uz;
        local[DIAGNOSTIC_SUM_TKE] += 0.5 * (varUx + varUy + varUz);
        local[DIAGNOSTIC_FLUID_COUNT] += 1.0;
    }

#pragma unroll
    for (natural_t field = 0; field < DIAGNOSTIC_FIELDS; ++field)
    {
        shared[field][thread] = local[field];
    }

    __syncthreads();

    for (natural_t offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (thread < offset)
        {
#pragma unroll
            for (natural_t field = 0; field < DIAGNOSTIC_FIELDS; ++field)
            {
                shared[field][thread] += shared[field][thread + offset];
            }
        }
        __syncthreads();
    }

    if (thread == 0)
    {
#pragma unroll
        for (natural_t field = 0; field < DIAGNOSTIC_FIELDS; ++field)
        {
            blockSums[blockIdx.x * DIAGNOSTIC_FIELDS + field] = shared[field][0];
        }
    }
}

static inline cudaError_t initKineticEnergyDiagnostics(
    KineticEnergyDiagnostics &diagnostics)
{
    diagnostics.hostBlockSums.assign(DIAGNOSTIC_BLOCKS * DIAGNOSTIC_FIELDS, 0.0);
    diagnostics.statisticsSamples = 0;

    cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&diagnostics.deviceBlockSums),
                                 diagnostics.hostBlockSums.size() * sizeof(double));
    if (err != cudaSuccess)
    {
        return err;
    }

    err = cudaMalloc(reinterpret_cast<void **>(&diagnostics.deviceStatistics),
                     static_cast<size_t>(DIAGNOSTIC_STATS_FIELDS) * static_cast<size_t>(CELLS) * sizeof(real_t));
    if (err != cudaSuccess)
    {
        cudaFree(diagnostics.deviceBlockSums);
        diagnostics.deviceBlockSums = nullptr;
        return err;
    }

    err = cudaMemset(diagnostics.deviceStatistics,
                     0,
                     static_cast<size_t>(DIAGNOSTIC_STATS_FIELDS) * static_cast<size_t>(CELLS) * sizeof(real_t));
    if (err != cudaSuccess)
    {
        cudaFree(diagnostics.deviceStatistics);
        cudaFree(diagnostics.deviceBlockSums);
        diagnostics.deviceStatistics = nullptr;
        diagnostics.deviceBlockSums = nullptr;
    }

    return err;
}

static inline cudaError_t destroyKineticEnergyDiagnostics(
    KineticEnergyDiagnostics &diagnostics)
{
    cudaError_t firstError = cudaFree(diagnostics.deviceBlockSums);
    const cudaError_t statsError = cudaFree(diagnostics.deviceStatistics);
    if (firstError == cudaSuccess)
    {
        firstError = statsError;
    }

    diagnostics.deviceBlockSums = nullptr;
    diagnostics.deviceStatistics = nullptr;
    diagnostics.hostBlockSums.clear();
    diagnostics.statisticsSamples = 0;
    return firstError;
}

static inline void openKineticEnergyDiagnosticOutput(
    KineticEnergyDiagnostics &diagnostics,
    const bool append)
{
    const std::filesystem::path dir("output/diagnostics");
    std::filesystem::create_directories(dir);

    const std::ios::openmode mode = std::ios::binary | (append ? std::ios::app : std::ios::trunc);

    diagnostics.keTotalOutput.open(dir / "ke_total.bin", mode);
    diagnostics.keMeanOutput.open(dir / "ke_mean.bin", mode);
    diagnostics.tkeOutput.open(dir / "tke.bin", mode);

    if (!diagnostics.keTotalOutput || !diagnostics.keMeanOutput || !diagnostics.tkeOutput)
    {
        std::cerr << "Could not open kinetic energy diagnostic output files" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static inline void closeKineticEnergyDiagnosticOutput(
    KineticEnergyDiagnostics &diagnostics)
{
    diagnostics.keTotalOutput.close();
    diagnostics.keMeanOutput.close();
    diagnostics.tkeOutput.close();
}

static inline void writeKineticEnergyDiagnosticRecord(
    std::ofstream &out,
    const natural_t step,
    const double value)
{
    out.write(reinterpret_cast<const char *>(&step), sizeof(step));
    out.write(reinterpret_cast<const char *>(&value), sizeof(value));

    if (!out)
    {
        std::cerr << "Could not write kinetic energy diagnostic output" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static inline void writeKineticEnergyDiagnostics(
    KineticEnergyDiagnostics &diagnostics,
    const natural_t step,
    const KineticEnergySample &sample)
{
    writeKineticEnergyDiagnosticRecord(diagnostics.keTotalOutput, step, sample.keTotal);
    writeKineticEnergyDiagnosticRecord(diagnostics.keMeanOutput, step, sample.keMean);
    writeKineticEnergyDiagnosticRecord(diagnostics.tkeOutput, step, sample.tke);
}

static inline cudaError_t updateKineticEnergyStatistics(
    KineticEnergyDiagnostics &diagnostics,
    const real_t *__restrict__ moments)
{
    const natural_t nextSample = diagnostics.statisticsSamples + 1;
    constexpr dim3 block(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
    constexpr dim3 grid(GRID_X, GRID_Y, GRID_Z);

    updateKineticEnergyStatisticsKernel<<<grid, block>>>(
        moments,
        diagnostics.deviceStatistics,
        nextSample);

    const cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess)
    {
        diagnostics.statisticsSamples = nextSample;
    }

    return err;
}

static inline cudaError_t computeKineticEnergyDiagnostics(
    KineticEnergyDiagnostics &diagnostics,
    const real_t *__restrict__ moments,
    KineticEnergySample &sample)
{
    kineticEnergyPartialSums<<<DIAGNOSTIC_BLOCKS, DIAGNOSTIC_THREADS>>>(
        moments,
        diagnostics.deviceStatistics,
        diagnostics.deviceBlockSums);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return err;
    }

    err = cudaMemcpy(diagnostics.hostBlockSums.data(),
                     diagnostics.deviceBlockSums,
                     diagnostics.hostBlockSums.size() * sizeof(double),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        return err;
    }

    double sums[DIAGNOSTIC_FIELDS] = {};
    for (natural_t block = 0; block < DIAGNOSTIC_BLOCKS; ++block)
    {
#pragma unroll
        for (natural_t field = 0; field < DIAGNOSTIC_FIELDS; ++field)
        {
            sums[field] += diagnostics.hostBlockSums[block * DIAGNOSTIC_FIELDS + field];
        }
    }

    sample = {};
    sample.fluidCells = static_cast<natural_t>(sums[DIAGNOSTIC_FLUID_COUNT]);
    sample.statisticsSamples = diagnostics.statisticsSamples;

    if (sample.fluidCells == 0)
    {
        return cudaSuccess;
    }

    const double invFluid = 1.0 / static_cast<double>(sample.fluidCells);

    sample.keTotal = 0.5 * (sums[DIAGNOSTIC_SUM_UX2] +
                            sums[DIAGNOSTIC_SUM_UY2] +
                            sums[DIAGNOSTIC_SUM_UZ2]);
    sample.keMean = sample.keTotal * invFluid;
    sample.tke = sums[DIAGNOSTIC_SUM_TKE] * invFluid;

    return cudaSuccess;
}
