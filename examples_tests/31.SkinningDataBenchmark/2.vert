// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "common.glsl"


layout(std430, set = 0, binding = 0, row_major) restrict readonly buffer BoneMatrices
{
    mat4 boneMatrix[MAT_MAX_CNT];
    mat4x3 normalMatrix[MAT_MAX_CNT];
};

#ifndef BENCHMARK
layout(location = 0) in vec3 pos;
layout(location = 3) in vec3 normal;
layout(location = 0) out vec3 vNormal;
#endif
layout(location = 4) in int boneID;


void main()
{
#ifdef BENCHMARK
    const vec3 pos = vec3(1.0, 2.0, 3.0);
    const vec3 normal = vec3(1.0, 2.0, 3.0);
#endif

#ifndef BENCHMARK    
    gl_Position = boneMatrix[boneID] * vec4(pos, 1.0);
    vNormal = mat3(normalMatrix[boneID]) * normalize(normal);
#else
    gl_Position = boneMatrix[boneID] * vec4(pos, 1.0);
    gl_Position.xyz += mat3(normalMatrix[boneID]) * normal;
#endif
}