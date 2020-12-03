// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifdef __cplusplus
    #define uint uint32_t
    #define mat4 core::matrix4SIMD
    #define mat3 core::matrix3x4SIMD
#endif

struct CullShaderData_t
{
    mat4 viewProjMatrix;
    mat3 viewInverseTransposeMatrix;
    uint maxDrawCount;
    uint cull;
};


#ifdef __cplusplus
    #undef uint
    #undef mat4
    #undef mat3
#endif

#define kCullWorkgroupSize 256