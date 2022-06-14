// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_SHAPES_RECTANGLE_INCLUDED_
#define _NBL_BUILTIN_GLSL_SHAPES_RECTANGLE_INCLUDED_

#include <nbl/builtin/glsl/math/functions.glsl>

vec3 nbl_glsl_shapes_getSphericalRectangle(in vec3 observer, in vec3 rectangleOrigin, in mat3 rectangleNormalBasis)
{
    return (rectangleOrigin-observer) * rectangleNormalBasis;
}

float nbl_glsl_shapes_SolidAngleOfRectangle(in vec3 r0, in vec2 rectangleExtents, out float b0, out float b1, out float k) 
{
    const vec4 denorm_n_z = vec4(-r0.y, r0.x+rectangleExtents.x, r0.y+rectangleExtents.y, -r0.x);
    const vec4 n_z = denorm_n_z*inversesqrt(vec4(r0.z*r0.z)+denorm_n_z*denorm_n_z);
    const vec4 cosGamma = vec4(
        -n_z[0]*n_z[1],
        -n_z[1]*n_z[2],
        -n_z[2]*n_z[3],
        -n_z[3]*n_z[0]
    );
    
    const vec4 g = acos(cosGamma);

    k = 2*nbl_glsl_PI - g[2] - g[3];
    b0 = n_z[0];
    b1 = n_z[2];
    return g[0] + g[1] + g[2] + g[3] - 2*nbl_glsl_PI;
}

#endif
