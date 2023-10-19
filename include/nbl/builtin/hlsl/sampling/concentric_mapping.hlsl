// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SAMPLING_CONCENTRIC_MAPPING_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_CONCENTRIC_MAPPING_INCLUDED_

#include <nbl/builtin/hlsl/math/functions.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

float2 concentricMapping(in float2 _u)
{
    //map [0;1]^2 to [-1;1]^2
    float2 u = 2.0f * _u - 1.0f;

    float2 p;
    if (all(u == float2(0.0,0.0)))
        p = float2(0.0,0.0);
    else
    {
        float r;
        float theta;
        if (abs(u.x) > abs(u.y)) {
            r = u.x;
            theta = 0.25f * math::PI * (u.y / u.x);
        }
        else {
            r = u.y;
            theta = 0.5f * math::PI - 0.25f * math::PI * (u.x / u.y);
        }
        // TODO: use nbl_glsl_sincos, but check that theta is in [-PI,PI]
        p = r * float2(cos(theta), sin(theta));
    }

    return p;
}

}
}
}

#endif