#version 460 core

#include "common.glsl"

layout(std430, set = 0, binding = 0, row_major) restrict readonly buffer BoneMatrices
{
    mat4 boneMatrix[MAT_MAX_CNT];
    mat4x3 normalMatrix[MAT_MAX_CNT];
};

layout(location = 0) in vec3 pos;
layout(location = 3) in vec3 normal;
layout(location = 4) in int boneID;

layout(location = 0) out vec3 vNormal;

void main()
{
    gl_Position = boneMatrix[boneID] * vec4(pos, 1.0);
    vNormal = normalMatrix[boneID] * vec4(normalize(normal), 0.0);
}