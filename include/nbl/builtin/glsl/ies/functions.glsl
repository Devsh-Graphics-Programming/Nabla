// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_GLSL_IES_FUNCTIONS_INCLUDED_
#define _NBL_GLSL_IES_FUNCTIONS_INCLUDED_

#include <nbl/builtin/glsl/math/constants.glsl>

// TODO: when rewriting to HLSL this is not IES namespace or folder, this should be octahedral mapping sitting somewhere where the spherical/polar sits
// NOTE: I changed it to return NDC [-1,1]^2 instead of UV coords [0,1]^2
vec2 nbl_glsl_TODOnamespace_octahedral_mapping(vec3 dir)
{
    float sum = dot(vec3(1.0f), abs(dir));
    vec3 s = dir / sum;    

    if(s.z < 0.0f)
    {
        const uvec2 flipSignMask = floatBitsToUint(s.xy)&0x80000000u;
        s.xy = uintBitsToFloat(floatBitsToUint(1.0f - abs(s.yx))^flipSignMask);
    }

    return s.xy;
}

// TODO: implement proper mirroing
// MIRROR_180_BITS = 0b001, Last Angle is 180, so map V with MIRROR and corner sampling off
// MIRROR_90_BITS = 0b010, Last Angle is 90, so map both U and V with MIRROR and corner sampling off
// ISOTROPIC_BITS = 0b011, texture to sample is Nx1, pretend v=middle always , and make u REPEAT or CLAMP_TO_BORDER
// FULL_THETA_BIT = 0b100, handle truncated domain and rotate by 45 degrees for anisotropic
// (certain combos wont work like 90 degree 2 symmetry domain & half theta), it really needs to be an 8 case label thing explicitly enumerated
vec2 nbl_glsl_IES_convert_dir_to_uv(vec3 dir, vec2 halfMinusHalfPixel)
{
    // halfMinusHalfPixel = 0.5-0.5/texSize
    // believe it or not, cornerSampled(NDC*0.5+0.5) = NDC*0.5*(1-1/texSize)+0.5
    return nbl_glsl_TODOnamespace_octahedral_mapping(dir)*halfMinusHalfPixel+0.5;
}

#endif