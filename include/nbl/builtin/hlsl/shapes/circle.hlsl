// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SHAPES_CIRCLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_CIRCLE_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace shapes
{
    struct Circle_t
    {
        float2 center;
        float radius;

        static Circle_t construct(float2 center, float radius)
        {
            Circle_t c = { center, radius };
            return c;
        }

        float signedDistance(float2 p)
        {
            return distance(p, center) - radius;
        }
    };
}
}
}

#endif