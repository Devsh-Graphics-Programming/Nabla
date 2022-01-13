// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _COMMON_GLSL_INCLUDED_
#define _COMMON_GLSL_INCLUDED_


#ifdef __cplusplus
#define uint uint32_t
#define vec3 struct{float comp[3];}
#define mat4 nbl::core::matrix4SIMD
#endif
struct PerViewPerInstance_t
{
    mat4 mvp;
    uint lod;
    uint padding0;
    uint padding1;
    uint padding2;
};

struct CullPushConstants_t
{
    mat4 viewProjMat;
    vec3 camPos;
    float fovDilationFactor;
    uint instanceCount;
};
#ifdef __cplusplus
#undef mat4
#undef vec3
#undef uint
#else
#define nbl_glsl_PerViewPerInstance_t PerViewPerInstance_t
#endif


#endif