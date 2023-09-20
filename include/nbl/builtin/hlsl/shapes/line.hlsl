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
    struct Line_t
    {
        float2 start;
        float2 end;
        float thickness;

        static Line_t construct(float2 start, float2 end, float thickness)
        {
            Line_t ret = { start, end, thickness };
            return ret;
        }

        float signedDistance(float2 p)
        {
            const float l = length(end - start);
            const float2  d = (end - start) / l;
            float2  q = p - (start + end) * 0.5;
            q = mul(float2x2(d.x, d.y, -d.y, d.x), q);
            q = abs(q) - float2(l * 0.5, thickness);
            return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
        }
    };
}
}
}

#endif