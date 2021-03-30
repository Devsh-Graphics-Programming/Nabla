#version 430 core
layout(location = 0) in vec4 vPos;
layout(location = 3) in vec3 vNormal;


layout( push_constant, row_major ) uniform Block {
	mat4 modelViewProj;
} PushConstants;

layout(location = 0) out vec3 normal; 

void main()
{
    gl_Position = PushConstants.modelViewProj*vPos;
    normal = vNormal;
}