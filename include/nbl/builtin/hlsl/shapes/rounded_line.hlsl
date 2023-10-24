// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SHAPES_ROUNDED_LINE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_ROUNDED_LINE_INCLUDED_

#include <nbl/builtin/hlsl/shapes/line.hlsl>
#include <nbl/builtin/hlsl/shapes/circle.hlsl>

namespace nbl
{
namespace hlsl
{
namespace shapes
{
    struct RoundedLine_t
    {
        float32_t2 start;
        float32_t2 end;
        float32_t thickness;

        static RoundedLine_t construct(float32_t2 start, float32_t2 end, float32_t thickness)
        {
            RoundedLine_t ret = { start, end, thickness };
            return ret;
        }

        float32_t signedDistance(float32_t2 p)
        {
            const float32_t startCircleSD = Circle_t::construct(start, thickness).signedDistance(p);
            const float32_t endCircleSD = Circle_t::construct(end, thickness).signedDistance(p);
            const float32_t lineSD = Line_t::construct(start, end, thickness).signedDistance(p);
            return min(lineSD, min(startCircleSD, endCircleSD));
        }
    };
}
}
}

#endif