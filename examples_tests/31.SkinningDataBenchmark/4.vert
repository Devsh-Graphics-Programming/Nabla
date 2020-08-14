#include "common.glsl"

    //TODO: max mat cnt for this shader

layout( push_constant, row_major ) uniform Block 
{
	uint matrixOffsets[16];
}pc;

layout(std430, set = 0, binding = 0, row_major) restrict readonly buffer BoneMatrices
{
    float boneMatComp[BONE_COMP_MAX_CNT];
    float normalMatComp[NORM_COMP_MAX_CNT];
};

#ifndef BENCHMARK
layout(location = 0) in vec3 pos;
layout(location = 3) in vec3 normal;
layout(location = 0) out vec3 vNormal;
#endif

layout(location = 4) in uint boneID;


void main()
{
    #ifdef BENCHMARK
    const vec3 pos = vec3(1.0, 2.0, 3.0);
    const vec3 normal = vec3(1.0, 2.0, 3.0);
    #endif
    
    mat4 mvp = mat4(
        boneMatComp[boneID + pc.matrixOffsets[0]], boneMatComp[boneID + pc.matrixOffsets[4]], boneMatComp[boneID + pc.matrixOffsets[8]],  boneMatComp[boneID + pc.matrixOffsets[12]],
        boneMatComp[boneID + pc.matrixOffsets[1]], boneMatComp[boneID + pc.matrixOffsets[5]], boneMatComp[boneID + pc.matrixOffsets[9]],  boneMatComp[boneID + pc.matrixOffsets[13]],
        boneMatComp[boneID + pc.matrixOffsets[2]], boneMatComp[boneID + pc.matrixOffsets[6]], boneMatComp[boneID + pc.matrixOffsets[10]], boneMatComp[boneID + pc.matrixOffsets[14]],
        boneMatComp[boneID + pc.matrixOffsets[3]], boneMatComp[boneID + pc.matrixOffsets[7]], boneMatComp[boneID + pc.matrixOffsets[11]], boneMatComp[boneID + pc.matrixOffsets[15]]
    );
    mat3 normalMatrix = mat3(
        normalMatComp[boneID + pc.matrixOffsets[0]], normalMatComp[boneID + pc.matrixOffsets[3]], normalMatComp[boneID + pc.matrixOffsets[6]],
        normalMatComp[boneID + pc.matrixOffsets[1]], normalMatComp[boneID + pc.matrixOffsets[4]], normalMatComp[boneID + pc.matrixOffsets[7]],
        normalMatComp[boneID + pc.matrixOffsets[2]], normalMatComp[boneID + pc.matrixOffsets[5]], normalMatComp[boneID + pc.matrixOffsets[8]]
    );

#ifndef BENCHMARK
    gl_Position = mvp * vec4(pos, 1.0);
    vNormal = normalMatrix * normalize(normal);
#else
    gl_Position = mvp * vec4(pos, 1.0);
    gl_Position.xyz += normalMatrix * normal;
#endif
}