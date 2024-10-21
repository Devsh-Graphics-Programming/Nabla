// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SHAPES_UTIL_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_UTIL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{
namespace shapes
{
namespace util
{

template<typename float_t>
static vector<float_t, 2> LineLineIntersection(NBL_CONST_REF_ARG(vector<float_t, 2>) P1, NBL_CONST_REF_ARG(vector<float_t, 2>) V1, NBL_CONST_REF_ARG(vector<float_t, 2>) P2, NBL_CONST_REF_ARG(vector<float_t, 2>) V2)
{
    typedef vector<float_t, 2> float_t2;

    float_t denominator = V1.y * V2.x - V1.x * V2.y;
    vector<float_t, 2> diff = P1 - P2;
    float_t numerator = dot(float_t2(V2.y, -V2.x), float_t2(diff.x, diff.y));

    if (abs(denominator) < 1e-15 && abs(numerator) < 1e-15)
    {
        // are parallel and the same
        return (P1 + P2) / 2.0;
    }

    float_t t = numerator / denominator;
    float_t2 intersectionPoint = P1 + t * V1;
    return intersectionPoint;
}

}
}
}
}

#endif