// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SHAPES_LINE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_LINE_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace shapes
{
    template<typename float_t>
    struct Line_t
    {
        using float2_t = vector<float_t, 2>;
        using float2x2_t = matrix<float_t, 2, 2>;

        float2_t start;
        float2_t end;

        static Line_t construct(float2_t start, float2_t end)
        {
            Line_t ret = { start, end };
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
        float_t signedDistance(float2_t p, float_t thickness, Clipper clipper = DefaultClipper::construct())
        {
            const float_t l = length(end - start);
            const float2_t  d = (end - start) / l;
            float2_t  q = p - (start + end) * 0.5;
            q = mul(float2x2_t(d.x, d.y, -d.y, d.x), q);
            q = abs(q) - float2_t(l * 0.5, thickness);
            return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
        }
    };
}
}
}

#endif