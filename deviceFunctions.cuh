#pragma once

#include "constants.cuh"

#include <utility>

__device__ __host__ [[nodiscard]] static __forceinline__ natural_t global3(
    const natural_t x,
    const natural_t y,
    const natural_t z) noexcept
{
    return x + y * NX + z * STRIDE;
}

__device__ __host__ [[nodiscard]] static __forceinline__ natural_t momentIdx(
    const natural_t field,
    const natural_t id) noexcept
{
    return id + CELLS * field;
}

__device__ __host__ [[nodiscard]] static __forceinline__ real_t &moment(
    real_t *moments,
    const natural_t field,
    const natural_t id) noexcept
{
    return moments[momentIdx(field, id)];
}

template <typename T, T v>
struct IntegralConstant
{
    static constexpr const T value = v;
    using value_type = T;
    using type = IntegralConstant;

    __device__ [[nodiscard]] inline consteval operator value_type() const noexcept
    {
        return value;
    }

    __device__ [[nodiscard]] inline consteval value_type operator()() const noexcept
    {
        return value;
    }
};

template <const natural_t Start, const natural_t End, typename F>
__device__ __forceinline__ constexpr void constexpr_for(F &&f) noexcept
{
    if constexpr (Start < End)
    {
        f(IntegralConstant<natural_t, Start>());
        if constexpr (Start + 1 < End)
        {
            constexpr_for<Start + 1, End>(std::forward<F>(f));
        }
    }
}

__device__ __host__ [[nodiscard]] static __forceinline__ const real_t &moment(
    const real_t *moments,
    const natural_t field,
    const natural_t id) noexcept
{
    return moments[momentIdx(field, id)];
}
