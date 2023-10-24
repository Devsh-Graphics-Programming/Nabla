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
        float32_t2 start;
        float32_t2 end;
        float32_t thickness;

        static Line_t construct(float32_t2 start, float32_t2 end, float32_t thickness)
        {
            Line_t ret = { start, end, thickness };
            return ret;
        }

        float32_t signedDistance(float32_t2 p)
        {
            const float32_t l = length(end - start);
            const float32_t2  d = (end - start) / l;
            float32_t2  q = p - (start + end) * 0.5;
            q = mul(float32_t2x2(d.x, d.y, -d.y, d.x), q);
            q = abs(q) - float32_t2(l * 0.5, thickness);
            return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
        }
    };
}
}
}

#endif