// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

struct ModelData_t
{
#ifdef __cplusplus
    core::matrix3x4SIMD worldMatrix;
    core::matrix3x4SIMD normalMatrix;
    core::vectorSIMDf   bbox[2];
#else
    mat4x3  worldMatrix;
    mat3    normalMatrix;
    vec3    bbox[2];
#endif
};

struct DrawData_t
{
#ifdef __cplusplus
    core::matrix4SIMD   modelViewProjMatrix;
    core::matrix3x4SIMD normalMatrix;
#else
    mat4 modelViewProjMatrix;
    mat3 normalMatrix;
#endif
};

#ifndef __cplusplus
#include <nbl/builtin/glsl/utils/common.glsl>
#include <nbl/builtin/glsl/utils/transform.glsl>

#include <nbl/builtin/glsl/utils/indirect_commands.glsl>

#include <nbl/builtin/glsl/utils/culling.glsl>
#endif