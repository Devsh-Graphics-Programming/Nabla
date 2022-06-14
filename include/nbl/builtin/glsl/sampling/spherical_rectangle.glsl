// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_SAMPLING_SPHERICAL_RECTANGLE_INCLUDED_
#define _NBL_BUILTIN_GLSL_SAMPLING_SPHERICAL_RECTANGLE_INCLUDED_

#include <nbl/builtin/glsl/math/quaternions.glsl>
#include <nbl/builtin/glsl/shapes/rectangle.glsl>

#define CLAMP_EPS 1e-5f

// Code from https://www.arnoldrenderer.com/research/egsr2013_spherical_rectangle.pdf
vec3 nbl_glsl_sampling_generateSphericalRectangleSample(in vec3 r0, in vec3 r1, in float S, in float b0, float b1, in float k, vec2 uv)
{
    // 1. compute ’cu’
    float au = uv.x * S + k;
    float fu = (cos(au) * b0 - b1) / sin(au);
    float cu = 1/sqrt(fu*fu + b0 * b0) * (fu>0 ? +1 : -1);
    cu = clamp(cu, -1, 1); // avoid NaNs
    // 2. compute ’xu’
    float xu = -(cu * r0.z) / sqrt(1 - cu*cu);
    xu = clamp(xu, r0.x, r1.x); // avoid Infs
    // 3. compute ’yv’
    float d = sqrt(xu*xu + r0.z*r0.z);
    float h0 = r0.y / sqrt(d*d + r0.y*r0.y);
    float h1 = r1.y / sqrt(d*d + r1.y*r1.y);
    float hv = h0 + uv.y * (h1-h0), hv2 = hv*hv;
    float yv = (hv2 < 1-CLAMP_EPS) ? (hv*d)/sqrt(1-hv2) : r1.y;
    // 4. transform (xu,yv,z0) to world coords
    return vec3(xu, yv, r0.z);
}

#endif