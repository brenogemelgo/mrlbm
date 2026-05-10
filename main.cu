#include "initialConditions.cuh"
#include "kernel.cuh"
#include "output.cuh"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <utility>

#define BENCHMARK

#define CUDA_CHECK(call)                                                         \
    do                                                                           \
    {                                                                            \
        const cudaError_t err = (call);                                          \
        if (err != cudaSuccess)                                                  \
        {                                                                        \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (false)

int main()
{
    real_t *moments = nullptr;
    real_t *dbuffer = nullptr;
    constexpr size_t bytes = static_cast<size_t>(NUM_MOMENTS) * static_cast<size_t>(CELLS) * sizeof(real_t);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&moments), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dbuffer), bytes));

    real_t *momentsAlloc = moments;
    real_t *dbufferAlloc = dbuffer;

    constexpr dim3 block(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
    constexpr dim3 grid(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z);

    cavityInit<<<grid, block>>>(moments, dbuffer);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto start = std::chrono::high_resolution_clock::now();

    for (natural_t t = 0; t < NSTEPS; ++t)
    {
        streamCollide<<<grid, block>>>(moments, dbuffer);
#ifndef BENCHMARK
        CUDA_CHECK(cudaGetLastError());
#endif
        std::swap(moments, dbuffer);

#ifndef BENCHMARK
        if ((t + 1) % STAMP == 0)
        {
            std::cout << "step " << (t + 1) << " / " << NSTEPS << std::endl;
            writeOutput(moments, t + 1);
        }
#endif
    }

#ifndef BENCHMARK
    CUDA_CHECK(cudaGetLastError());
#endif
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    const double mlups = static_cast<double>(CELLS) * static_cast<double>(NSTEPS) / elapsed.count() / static_cast<double>(1000000);

    std::cout << "elapsed: " << elapsed.count() << " s" << std::endl;
    std::cout << "MLUPS: " << mlups << std::endl;

    CUDA_CHECK(cudaFree(momentsAlloc));
    CUDA_CHECK(cudaFree(dbufferAlloc));
    return 0;
}
