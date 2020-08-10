#version 460 core

#include "common.glsl"

struct BoneNormalMatPair
{
    mat4 boneMatrix;
    mat4x3 normalMatrix;
};

layout(std430, set = 0, binding = 0, row_major) restrict readonly buffer BoneMatrices
{
    BoneNormalMatPair matrices[];
};

#ifndef BENCHMARK
layout(location = 0) in vec3 pos;
layout(location = 3) in vec3 normal;
#endif
layout(location = 4) in uint boneID;

layout(location = 0) out vec3 vNormal;

void main()
{
    #ifdef BENCHMARK
    const vec3 pos = vec3(1.0, 2.0, 3.0);
    const vec3 normal = vec3(1.0, 2.0, 3.0);
    #endif

    gl_Position = matrices[boneID].boneMatrix * vec4(pos, 1.0);
    //vNormal = vec3(matrices[boneID].normalMatrix * vec4(normalize(normal), 0.0));
    vNormal = mat3(matrices[boneID].normalMatrix) * normalize(normal);
}