#include "initialConditions.cuh"
#include "kernel.cuh"

#include <chrono>
#include <cstdlib>
#include <iostream>

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
    constexpr size_t bytes = static_cast<size_t>(NUM_FIELDS) * static_cast<size_t>(CELLS) * sizeof(real_t);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&moments), bytes));

    const dim3 block(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
    const dim3 grid(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z);

    cavityInit<<<grid, block>>>(moments);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto start = std::chrono::high_resolution_clock::now();

    for (natural_t t = 0; t < NSTEPS; ++t)
    {
        streamCollide<<<grid, block>>>(moments);

        if ((t + 1) % STAMP == 0)
        {
            std::cout << "step " << (t + 1) << " / " << NSTEPS << std::endl;
        }
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    const double mlups = static_cast<double>(CELLS) * static_cast<double>(NSTEPS) / elapsed.count() / static_cast<double>(1000000);

    std::cout << "elapsed: " << elapsed.count() << " s" << std::endl;
    std::cout << "MLUPS: " << mlups << std::endl;

    CUDA_CHECK(cudaFree(moments));
    return 0;
}
