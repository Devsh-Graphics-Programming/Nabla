// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_MORTON_INCLUDED_
#define _NBL_BUILTIN_HLSL_MORTON_INCLUDED_

#ifdef __HLSL_VERSION
#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#else
#include <cstdint>
#endif

namespace nbl
{
namespace core
{

namespace impl
{

#ifdef __HLSL_VERSION
template <typename T>
T morton2d_mask(uint16_t _n) const
{
    const static uint64_t mask[5] =
    {
        0x5555555555555555ull,
        0x3333333333333333ull,
        0x0F0F0F0F0F0F0F0Full,
        0x00FF00FF00FF00FFull,
        0x0000FFFF0000FFFFull
    };
    return static_cast<T>(mask[_n]);
}

template <typename T>
T morton3d_mask(uint16_t _n) const
{
    const static uint64_t mask[5] =
    {
        0x1249249249249249ull,
        0x10C30C30C30C30C3ull,
        0x010F00F00F00F00Full,
        0x001F0000FF0000FFull,
        0x001F00000000FFFFull
    };
    return static_cast<T>(mask[_n]);
}
template <typename T>
T morton4d_mask(uint16_t _n) const
{
    const static uint64_t mask[4] =
    {
        0x1111111111111111ull,
        0x0303030303030303ull,
        0x000F000F000F000Full,
        0x000000FF000000FFull
    };
    return static_cast<T>(mask[_n]);
}

template <typename T, uint32_t bitDepth>
inline T morton2d_decode(T x)
{
    x = x & morton2d_mask<T>(0);
    x = (x | (x >> 1)) & morton2d_mask<T>(1);
    x = (x | (x >> 2)) & morton2d_mask<T>(2);
    if (bitDepth > 8u)
    {
        x = (x | (x >> 4)) & morton2d_mask<T>(3);
    }
    if (bitDepth > 16u)
    {
        x = (x | (x >> 8)) & morton2d_mask<T>(4);
    }
    if (bitDepth > 32u)
    {
        x = (x | (x >> 16));
    }
    return x;
}

//! Puts bits on even positions filling gaps with 0s
template <typename T, uint32_t bitDepth>
inline T separate_bits_2d(T x)
{
    if (bitDepth > 32u)
    {
        x = (x | (x << 16)) & morton2d_mask<T>(4);
    }
    if (bitDepth > 16u)
    {
        x = (x | (x << 8)) & morton2d_mask<T>(3);
    }
    if (bitDepth > 8u)
    {
        x = (x | (x << 4)) & morton2d_mask<T>(2);
    }
    x = (x | (x << 2)) & morton2d_mask<T>(1);
    x = (x | (x << 1)) & morton2d_mask<T>(0);

    return x;
}
template <typename T, uint32_t bitDepth>
inline T separate_bits_3d(T x)
{
    if (bitDepth > 32u)
    {
        x = (x | (x << 32)) & morton3d_mask<T>(4);
    }
    if (bitDepth > 16u)
    {
        x = (x | (x << 16)) & morton3d_mask<T>(3);
    }
    if (bitDepth > 8u)
    {
        x = (x | (x << 8)) & morton3d_mask<T>(2);
    }
    x = (x | (x << 4)) & morton3d_mask<T>(1);
    x = (x | (x << 2)) & morton3d_mask<T>(0);

    return x;
}
template <typename T, uint32_t bitDepth>
inline T separate_bits_4d(T x)
{
    if (bitDepth > 32u)
    {
        x = (x | (x << 24)) & morton4d_mask<T>(3);
    }
    if (bitDepth > 16u)
    {
        x = (x | (x << 12)) & morton4d_mask<T>(2);
    }
    if (bitDepth > 8u)
    {
        x = (x | (x << 6)) & morton4d_mask<T>(1);
    }
    x = (x | (x << 3)) & morton4d_mask<T>(0);

    return x;
}
#else
template <typename T>
constexpr T morton2d_mask(uint8_t _n)
{
    constexpr uint64_t mask[5] =
    {
        0x5555555555555555ull,
        0x3333333333333333ull,
        0x0F0F0F0F0F0F0F0Full,
        0x00FF00FF00FF00FFull,
        0x0000FFFF0000FFFFull
    };
    return static_cast<T>(mask[_n]);
}
template <typename T>
constexpr T morton3d_mask(uint8_t _n)
{
    constexpr uint64_t mask[5] =
    {
        0x1249249249249249ull,
        0x10C30C30C30C30C3ull,
        0x010F00F00F00F00Full,
        0x001F0000FF0000FFull,
        0x001F00000000FFFFull
    };
    return static_cast<T>(mask[_n]);
}
template <typename T>
constexpr T morton4d_mask(uint8_t _n)
{
    constexpr uint64_t mask[4] =
    {
        0x1111111111111111ull,
        0x0303030303030303ull,
        0x000F000F000F000Full,
        0x000000FF000000FFull
    };
    return static_cast<T>(mask[_n]);
}

template <typename T, uint32_t bitDepth>
inline T morton2d_decode(T x)
{
    x = x & morton2d_mask<T>(0);
    x = (x | (x >> 1)) & morton2d_mask<T>(1);
    x = (x | (x >> 2)) & morton2d_mask<T>(2);
    if constexpr (bitDepth > 8u)
    {
        x = (x | (x >> 4)) & morton2d_mask<T>(3);
    }
    if constexpr (bitDepth > 16u)
    {
        x = (x | (x >> 8)) & morton2d_mask<T>(4);
    }
    if constexpr (bitDepth > 32u)
    {
        x = (x | (x >> 16));
    }
    return x;
}

//! Puts bits on even positions filling gaps with 0s
template <typename T, uint32_t bitDepth>
inline T separate_bits_2d(T x)
{
    if constexpr (bitDepth > 32u)
    {
        x = (x | (x << 16)) & morton2d_mask<T>(4);
    }
    if constexpr (bitDepth > 16u)
    {
        x = (x | (x << 8)) & morton2d_mask<T>(3);
    }
    if constexpr (bitDepth > 8u)
    {
        x = (x | (x << 4)) & morton2d_mask<T>(2);
    }
    x = (x | (x << 2)) & morton2d_mask<T>(1);
    x = (x | (x << 1)) & morton2d_mask<T>(0);

    return x;
}
template <typename T, uint32_t bitDepth>
inline T separate_bits_3d(T x)
{
    if constexpr (bitDepth > 32u)
    {
        x = (x | (x << 32)) & morton3d_mask<T>(4);
    }
    if constexpr (bitDepth > 16u)
    {
        x = (x | (x << 16)) & morton3d_mask<T>(3);
    }
    if constexpr (bitDepth > 8u)
    {
        x = (x | (x << 8)) & morton3d_mask<T>(2);
    }
    x = (x | (x << 4)) & morton3d_mask<T>(1);
    x = (x | (x << 2)) & morton3d_mask<T>(0);

    return x;
}
template <typename T, uint32_t bitDepth>
inline T separate_bits_4d(T x)
{
    if constexpr (bitDepth > 32u)
    {
        x = (x | (x << 24)) & morton4d_mask<T>(3);
    }
    if constexpr (bitDepth > 16u)
    {
        x = (x | (x << 12)) & morton4d_mask<T>(2);
    }
    if constexpr (bitDepth > 8u)
    {
        x = (x | (x << 6)) & morton4d_mask<T>(1);
    }
    x = (x | (x << 3)) & morton4d_mask<T>(0);

    return x;
}
#endif
}

template<typename T, uint32_t bitDepth = sizeof(T) * 8u>
T morton2d_decode_x(T _morton) { return impl::morton2d_decode<T, bitDepth>(_morton); }
template<typename T, uint32_t bitDepth = sizeof(T) * 8u>
T morton2d_decode_y(T _morton) { return impl::morton2d_decode<T, bitDepth>(_morton >> 1); }

template<typename T, uint32_t bitDepth = sizeof(T) * 8u>
T morton2d_encode(T x, T y) { return impl::separate_bits_2d<T, bitDepth>(x) | (impl::separate_bits_2d<T, bitDepth>(y) << 1); }
template<typename T, uint32_t bitDepth = sizeof(T) * 8u>
T morton3d_encode(T x, T y, T z) { return impl::separate_bits_3d<T, bitDepth>(x) | (impl::separate_bits_3d<T, bitDepth>(y) << 1) | (impl::separate_bits_3d<T, bitDepth>(z) << 2); }
template<typename T, uint32_t bitDepth = sizeof(T) * 8u>
T morton4d_encode(T x, T y, T z, T w) { return impl::separate_bits_4d<T, bitDepth>(x) | (impl::separate_bits_4d<T, bitDepth>(y) << 1) | (impl::separate_bits_4d<T, bitDepth>(z) << 2) | (impl::separate_bits_4d<T, bitDepth>(w) << 3); }

}
}

#endif