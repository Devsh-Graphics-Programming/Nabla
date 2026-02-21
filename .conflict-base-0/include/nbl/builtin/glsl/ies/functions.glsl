// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_GLSL_IES_FUNCTIONS_INCLUDED_
#define _NBL_GLSL_IES_FUNCTIONS_INCLUDED_

#include <nbl/builtin/glsl/math/constants.glsl>

// TODO: implement proper mirroing
// MIRROR_180_BITS = 0b001, Last Angle is 180, so map V slightly differently
// MIRROR_90_BITS = 0b010, Last Angle is 90, so map both U and V slightly differently
// ISOTROPIC_BITS = 0b011, texture to sample is Nx1, pretend v=middle always  
// FULL_THETA_BIT = 0b100, handle extended domain and rotate by 45 degrees for anisotropic

vec2 nbl_glsl_IES_convert_dir_to_uv(vec3 dir) {
    float sum = dot(vec3(1.0f), abs(dir));        
    vec3 s = dir / sum;    

    if(s.z < 0.0f) {
        s.xy = sign(s.xy) * (1.0f - abs(s.yx));
    }

    return s.xy * 0.5f + 0.5f;
}

// vec2 nbl_glsl_IES_convert_dir_to_uv(vec3 dir) {
// 	return vec2((atan(dir.x, dir.y) + nbl_glsl_PI) / (2.0*nbl_glsl_PI), acos(dir.z) / nbl_glsl_PI);
// }

#endif