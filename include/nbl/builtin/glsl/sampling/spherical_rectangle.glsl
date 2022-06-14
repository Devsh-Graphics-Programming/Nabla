// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_SAMPLING_SPHERICAL_RECTANGLE_INCLUDED_
#define _NBL_BUILTIN_GLSL_SAMPLING_SPHERICAL_RECTANGLE_INCLUDED_

#include <nbl/builtin/glsl/math/quaternions.glsl>
#include <nbl/builtin/glsl/shapes/rectangle.glsl>

// Code from https://www.arnoldrenderer.com/research/egsr2013_spherical_rectangle.pdf
vec2 nbl_glsl_sampling_generateSphericalRectangleSample(vec3 r0, in vec2 rectangleExtents, in vec2 uv, out float S)
{
    const vec4 denorm_n_z = vec4(-r0.y, r0.x+rectangleExtents.x, r0.y+rectangleExtents.y, -r0.x);
    const vec4 n_z = denorm_n_z*inversesqrt(vec4(r0.z*r0.z)+denorm_n_z*denorm_n_z);
    const vec4 cosGamma = vec4(
        -n_z[0]*n_z[1],
        -n_z[1]*n_z[2],
        -n_z[2]*n_z[3],
        -n_z[3]*n_z[0]
    );
    
    float p = nbl_glsl_getSumofArccosAB(cosGamma[0], cosGamma[1]);
    float q = nbl_glsl_getSumofArccosAB(cosGamma[2], cosGamma[3]);

    const float k = 2*nbl_glsl_PI - q;
    const float b0 = n_z[0];
    const float b1 = n_z[2];
    S = p + q - 2*nbl_glsl_PI;

    const float CLAMP_EPS = 1e-5f;

    // flip z axsis if r0.z > 0
    const uint zFlipMask = (floatBitsToUint(r0.z)^0x80000000u)&0x80000000u;
    r0.z = uintBitsToFloat(floatBitsToUint(r0.z)^zFlipMask);
    vec3 r1 = r0 + vec3(rectangleExtents.x, rectangleExtents.y, 0);
    
    const float au = uv.x * S + k;
    const float fu = (cos(au) * b0 - b1) / sin(au);
    const float cu_2 = max(fu*fu+b0*b0,1.f); // forces `cu` to be in [-1,1]
    const float cu = uintBitsToFloat(floatBitsToUint(inversesqrt(cu_2))^(floatBitsToUint(fu)&0x80000000u));
    
    float xu = -(cu * r0.z) * inversesqrt(1 - cu*cu);
    xu = clamp(xu, r0.x, r1.x); // avoid Infs
    const float d_2 = xu*xu + r0.z*r0.z;
    const float d = sqrt(d_2);

    const float h0 = r0.y * inversesqrt(d_2 + r0.y*r0.y);
    const float h1 = r1.y * inversesqrt(d_2 + r1.y*r1.y);
    const float hv = h0 + uv.y * (h1-h0), hv2 = hv*hv;
    const float yv = (hv2 < 1-CLAMP_EPS) ? (hv*d)*inversesqrt(1-hv2) : r1.y;

    return vec2((xu-r0.x)/rectangleExtents.x, (yv-r0.y)/rectangleExtents.y);
}

#endif