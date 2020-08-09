#version 460 core

layout( push_constant, row_major ) uniform Block 
{
	uint matrixOffsets[16];
}pc;

layout(std430, set = 0, binding = 0, row_major) restrict readonly buffer BoneMatrices
{
    float matComp[];
};

layout(location = 0) in vec3 pos;
layout(location = 3) in vec3 normal;
layout(location = 4) in int boneID;

layout(location = 0) out vec3 vNormal;

void main()
{
    mat4 mvp = mat4(
        matComp[boneID + pc.matrixOffsets[0]], matComp[boneID + pc.matrixOffsets[4]], matComp[boneID + pc.matrixOffsets[8]],  matComp[boneID + pc.matrixOffsets[12]],
        matComp[boneID + pc.matrixOffsets[1]], matComp[boneID + pc.matrixOffsets[5]], matComp[boneID + pc.matrixOffsets[9]],  matComp[boneID + pc.matrixOffsets[13]],
        matComp[boneID + pc.matrixOffsets[2]], matComp[boneID + pc.matrixOffsets[6]], matComp[boneID + pc.matrixOffsets[10]], matComp[boneID + pc.matrixOffsets[14]],
        matComp[boneID + pc.matrixOffsets[3]], matComp[boneID + pc.matrixOffsets[7]], matComp[boneID + pc.matrixOffsets[11]], matComp[boneID + pc.matrixOffsets[15]]
    );
    gl_Position = mvp * vec4(pos, 1.0);
    vNormal = vec3(1.0);
}