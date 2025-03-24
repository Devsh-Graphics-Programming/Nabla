// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MATH_MORTON_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_MORTON_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

namespace nbl
{
namespace hlsl
{
namespace math
{

namespace impl
{

template<typename T, uint32_t bitDepth>
struct MortonComponent;

template<typename T>
struct MortonComponent<T, 8u>
{
    static T decode2d(T x)
    {
        x &= 0x55555555u;
        x = (x ^ (x >>  1u)) & 0x33333333u;
        x = (x ^ (x >>  2u)) & 0x0f0f0f0fu;
        x = (x ^ (x >>  4u)) & 0x00ff00ffu;
        return x;
    }
};

template<typename T>
struct MortonComponent<T, 32u>
{
    static T decode2d(T x)
    {
        x &= 0x55555555u;
        x = (x ^ (x >>  1u)) & 0x33333333u;
        x = (x ^ (x >>  2u)) & 0x0f0f0f0fu;
        x = (x ^ (x >>  4u)) & 0x00ff00ffu;
        x = (x ^ (x >>  8u)) & 0x0000ffffu;
        x = (x ^ (x >>  16u));
        return x;
    }
};

}

template<typename T, uint32_t bitDepth=sizeof(T)*8u>
struct Morton
{
    using vector2_type = vector<T, 2>;
    using component_type = impl::MortonComponent<T, bitDepth>;

    static vector2_type decode2d(T x)
    {
        return vector2_type(component_type::decode2d(x), component_type::decode2d(x >> 1u));
    }
};

}
}
}

#endif
