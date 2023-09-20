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
        float2 start;
        float2 end;
        float thickness;

        static RoundedLine_t construct(float2 start, float2 end, float thickness)
        {
            RoundedLine_t ret = { start, end, thickness };
            return ret;
        }

        float signedDistance(float2 p)
        {
            const float startCircleSD = Circle_t::construct(start, thickness).signedDistance(p);
            const float endCircleSD = Circle_t::construct(end, thickness).signedDistance(p);
            const float lineSD = Line_t::construct(start, end, thickness).signedDistance(p);
            return min(lineSD, min(startCircleSD, endCircleSD));
        }
    };
}
}
}

#endif