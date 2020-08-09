#version 460 core

layout( push_constant, row_major ) uniform Block 
{
    uvec4 matrixOffsets;
    uint normalMatrixArrayOffset;
}pc;

layout(std430, set = 0, binding = 0, row_major) restrict readonly buffer BoneMatrices
{
    vec4 row[];
};

layout(location = 0) in vec3 pos;
layout(location = 3) in vec3 normal;
layout(location = 4) in int boneID;

layout(location = 0) out vec3 vNormal;

void main()
{
    vec4 worldPos = vec4(
        dot(row[boneID + pc.matrixOffsets.x], vec4(pos, 1.0)),
        dot(row[boneID + pc.matrixOffsets.y], vec4(pos, 1.0)),
        dot(row[boneID + pc.matrixOffsets.z], vec4(pos, 1.0)),
        dot(row[boneID + pc.matrixOffsets.w], vec4(pos, 1.0))
    );
    gl_Position = worldPos;
    
    vNormal = vec3(
        dot(vec3(row[pc.normalMatrixArrayOffset + boneID + pc.matrixOffsets.x]), normal),
        dot(vec3(row[pc.normalMatrixArrayOffset + boneID + pc.matrixOffsets.y]), normal),
        dot(vec3(row[pc.normalMatrixArrayOffset + boneID + pc.matrixOffsets.z]), normal)
    );
}