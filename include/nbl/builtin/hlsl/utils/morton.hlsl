
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_UTILS_MORTON_INCLUDED_
#define _NBL_BUILTIN_HLSL_UTILS_MORTON_INCLUDED_


namespace nbl
{
namespace hlsl
{
namespace morton
{


uint decode2d4bComponent(in uint x) 
{
    x &= 0x55555555u;
    x = (x ^ (x >>  1u)) & 0x33333333u;
    x = (x ^ (x >>  2u)) & 0x0f0f0f0fu;
    return x;
}

uint decode2d8bComponent(in uint x) 
{
    x &= 0x55555555u;
    x = (x ^ (x >>  1u)) & 0x33333333u;
    x = (x ^ (x >>  2u)) & 0x0f0f0f0fu;
    x = (x ^ (x >>  4u)) & 0x00ff00ffu;
    return x;
}

uint2 decode2d4b(in uint x)
{
    return uint2(decode2d4bComponent(x), decode2d4bComponent(x >> 1u));
}

uint2 decode2d8b(in uint x)
{
    return uint2(decode2d8bComponent(x), decode2d8bComponent(x >> 1u));
}


}
}
}

#endif