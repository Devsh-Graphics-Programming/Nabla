#version 460 core

#include "common.glsl"

layout( push_constant, row_major ) uniform Block 
{
    uvec4 matrixOffsets;
}pc;

layout(std430, set = 0, binding = 0, row_major) restrict readonly buffer BoneMatrices
{
    vec4 boneMatrixRow[BONE_VEC_MAX_CNT];
    vec4 normalMatrixRow[NORM_VEC_MAX_CNT];
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
    
    vec4 worldPos = vec4(
        dot(boneMatrixRow[boneID + pc.matrixOffsets.x], vec4(pos, 1.0)),
        dot(boneMatrixRow[boneID + pc.matrixOffsets.y], vec4(pos, 1.0)),
        dot(boneMatrixRow[boneID + pc.matrixOffsets.z], vec4(pos, 1.0)),
        dot(boneMatrixRow[boneID + pc.matrixOffsets.w], vec4(pos, 1.0))
    );
    gl_Position = worldPos;
    
#ifndef BENCHMARK
    vNormal = vec3(
        dot(vec3(normalMatrixRow[boneID + pc.matrixOffsets.x]), normalize(normal)),
        dot(vec3(normalMatrixRow[boneID + pc.matrixOffsets.y]), normalize(normal)),
        dot(vec3(normalMatrixRow[boneID + pc.matrixOffsets.z]), normalize(normal))
    );
#else
    gl_Position.xyz = vec3(
        dot(vec3(normalMatrixRow[boneID + pc.matrixOffsets.x]), normal),
        dot(vec3(normalMatrixRow[boneID + pc.matrixOffsets.y]), normal),
        dot(vec3(normalMatrixRow[boneID + pc.matrixOffsets.z]), normal)
    );
#endif
}