// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_MORTON_H_INCLUDED__
#define __NBL_CORE_MORTON_H_INCLUDED__

#include <cstdint>
#include "nbl/macros.h"

namespace nbl
{
namespace core
{

namespace impl
{
    template <typename T>
    constexpr T morton2d_mask(uint32_t _n)
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

    template <typename T, uint32_t bitDepth>
    inline T morton2d_decode(T x)
    {
        x = x & morton2d_mask<T>(0);
        x = (x | (x >> 1)) & morton2d_mask<T>(1);
        x = (x | (x >> 2)) & morton2d_mask<T>(2);
        if constexpr (bitDepth>8u)
        {
            x = (x | (x >> 4)) & morton2d_mask<T>(3);
        }
        if constexpr (bitDepth>16u)
        {
            x = (x | (x >> 8)) & morton2d_mask<T>(4);
        }
        if constexpr (bitDepth>32u)
        {
            x = (x | (x >> 16));
        }
        return x;
    }

    //! Puts bits on even positions filling gaps with 0s
    template <typename T, uint32_t bitDepth>
    inline T separate_bits_2d(T x)
    {
        if constexpr (bitDepth>32u)
        {
            x = (x | (x << 16)) & morton2d_mask<T>(4);
        }
        if constexpr (bitDepth > 16u)
        {
            x = (x | (x << 8)) & morton2d_mask<T>(3);
        }
        if constexpr (bitDepth>8u)
        {
            x = (x | (x << 4)) & morton2d_mask<T>(2);
        }
        x = (x | (x << 2)) & morton2d_mask<T>(1);
        x = (x | (x << 1)) & morton2d_mask<T>(0);

        return x;
    }

    inline uint64_t separate_bits_3d(uint64_t x)
    {
        x &= 0x00000000001fffff;
        x = (x | x << 32) & 0x001f00000000ffff;
        x = (x | x << 16) & 0x001f0000ff0000ff;
        x = (x | x << 8) & 0x010f00f00f00f00f;
        x = (x | x << 4) & 0x10c30c30c30c30c3;
        x = (x | x << 2) & 0x1249249249249249;
        return x;
    }
}

template<typename T, uint32_t bitDepth=sizeof(T)*8u>
T morton2d_decode_x(T _morton) { return impl::morton2d_decode<T,bitDepth>(_morton); }
template<typename T, uint32_t bitDepth=sizeof(T)*8u>
T morton2d_decode_y(T _morton) { return impl::morton2d_decode<T,bitDepth>(_morton>>1); }

template<typename T, uint32_t bitDepth=sizeof(T)*8u>
T morton2d_encode(T x, T y) { return impl::separate_bits_2d<T,bitDepth>(x) | (impl::separate_bits_2d<T,bitDepth>(y)<<1); }

inline uint64_t morton3d_encode(uint64_t x, uint64_t y, uint64_t z) { return impl::separate_bits_3d(x) | (impl::separate_bits_3d(y) << 1) | (impl::separate_bits_3d(z) << 2); }

}}

#endif