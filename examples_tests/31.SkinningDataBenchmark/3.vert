#version 460 core

layout( push_constant, row_major ) uniform Block 
{
    mat4 vp;
    uvec4 modelMatrixOffset;
    uvec4 normalMatrixOffset;
	
}pc;

layout(std430, set = 0, binding = 0, row_major) restrict readonly buffer BoneMatrices
{
    vec3 column[];
};

layout(location = 0) in vec3 pos;
layout(location = 3) in vec3 normal;
layout(location = 4) in int boneID;

layout(location = 1) out vec3 worldPos;
layout(location = 2) out vec3 vNormal;

void main()
{
    mat4 m = mat4(
        vec4(column[boneID + pc.modelMatrixOffset.x], 0.0), 
        vec4(column[boneID + pc.modelMatrixOffset.y], 0.0), 
        vec4(column[boneID + pc.modelMatrixOffset.z], 0.0), 
        vec4(column[boneID + pc.modelMatrixOffset.w], 1.0));

    gl_Position = pc.vp * m * vec4(pos, 1.0);
    vNormal = normalize(normal);
    worldPos = pos;
}