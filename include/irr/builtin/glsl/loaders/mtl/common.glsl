// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _IRR_BUILTIN_GLSL_MTL_LOADER_COMMON_INCLUDED_
#define _IRR_BUILTIN_GLSL_MTL_LOADER_COMMON_INCLUDED_

struct irr_glsl_MTLMaterialParameters
{
    vec3 Ka;
    vec3 Kd;
    vec3 Ks;
    vec3 Ke;
    vec4 Tf;//w component doesnt matter
    float Ns;
    float d;
    float bm;
    float Ni;
    float roughness;
    float metallic;
    float sheen;
    float clearcoatThickness;
    float clearcoatRoughness;
    float anisotropy;
    float anisoRotation;
    //extra info
    uint extra;
};

#endif