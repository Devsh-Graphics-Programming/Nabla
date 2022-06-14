// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_SHAPES_RECTANGLE_INCLUDED_
#define _NBL_BUILTIN_GLSL_SHAPES_RECTANGLE_INCLUDED_

#include <nbl/builtin/glsl/math/functions.glsl>

struct SphRect {
    mat3 basis;
    vec3 r0;
    vec3 r1;
};

SphRect nbl_glsl_shapes_getSphericalRectangle(in vec3 start, in vec3 ex, in vec3 ey, in vec3 origin) 
{
    const float exl = length(ex);
    const float eyl = length(ey);
    const vec3 x = ex / exl;
    const vec3 y = ey / eyl;
    vec3 z = normalize(cross(x, y));
    if (dot(start-origin, z) > 0) {
        z*=-1;
    }

    const mat3 basis = mat3(x, y, z);
    const vec3 r0 = (start-origin) * basis;
    const vec3 r1 = r0 + vec3(exl, eyl, 0);

    SphRect rect;
    rect.basis = basis;
    rect.r0 = r0;
    rect.r1 = r1;
    return rect;
}

float nbl_glsl_shapes_SolidAngleOfRectangle(in vec3 r0, in vec3 r1, out float b0, out float b1, out float k) 
{
    const vec3 v00 = vec3(r0.x, r0.y, r0.z);
    const vec3 v01 = vec3(r0.x, r1.y, r0.z);
    const vec3 v10 = vec3(r1.x, r0.y, r0.z);
    const vec3 v11 = vec3(r1.x, r1.y, r0.z);

    const vec3 n0 = normalize(cross(v00, v10));
    const vec3 n1 = normalize(cross(v10, v11));
    const vec3 n2 = normalize(cross(v11, v01));
    const vec3 n3 = normalize(cross(v01, v00));

    const float g0 = acos(-n0.z * n1.z);
    const float g1 = acos(-n1.z * n2.z);
    const float g2 = acos(-n2.z * n3.z);
    const float g3 = acos(-n3.z * n0.z);

    k = 2*nbl_glsl_PI - g2 - g3;
    b0 = n0.z;
    b1 = n2.z;

    return g0 + g1 + g2 + g3 - 2*nbl_glsl_PI;
}

#endif
