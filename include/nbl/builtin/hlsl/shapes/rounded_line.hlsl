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
    template<typename float_t>
    struct RoundedLine_t
    {
        using float2_t = vector<float_t, 2>;

        float2_t start;
        float2_t end;

        static RoundedLine_t construct(float2_t start, float2_t end)
        {
            RoundedLine_t ret = { start, end };
            return ret;
        }

        struct DefaultClipper
        {
            static DefaultClipper construct()
            {
                DefaultClipper ret;
                return ret;
            }
        
            inline float2_t operator()(const float_t t)
            {
                const float_t ret = clamp(t, 0.0, 1.0);
                return float2_t(ret, ret);
            }
        };

        template<typename Clipper = DefaultClipper>
        float signedDistance(float2_t p, float_t thickness, Clipper clipper = DefaultClipper::construct())
        {
            const float_t startCircleSD = Circle_t::construct(start, thickness).signedDistance(p);
            const float_t endCircleSD = Circle_t::construct(end, thickness).signedDistance(p);
            const float_t lineSD = Line_t::construct(start, end, thickness).signedDistance(p);
            return min(lineSD, min(startCircleSD, endCircleSD));
        }
    };
}
}
}

#endif