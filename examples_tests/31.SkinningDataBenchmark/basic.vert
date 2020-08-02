#version 460 core

layout(std430, set = 0, binding = 0, row_major) restrict readonly buffer BoneMatrices
{
    mat4 matrix[];
};

layout( push_constant, row_major ) uniform Block 
{
	mat4 vp;
} 
pushConstants;

layout(location = 0) in vec3 pos;
layout(location = 3) in vec3 normal;

layout(location = 1) out vec3 worldPos;
layout(location = 2) out vec3 vNormal;

void main()
{    
    gl_Position = pushConstants.vp * matrix[0] * vec4(pos, 1.0);
    vNormal = normalize(normal);
    worldPos = pos;
}