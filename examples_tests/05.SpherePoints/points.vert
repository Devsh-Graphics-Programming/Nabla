#version 430 core
layout(location = 0) in vec3 vPos;

layout( push_constant, row_major ) uniform Block {
	mat4 modelViewProj;
} PushConstants;

layout(location = 0) out vec4 Color; //per vertex output color, will be interpolated across the triangle

void main()
{
    gl_Position = PushConstants.modelViewProj*vec4(normalize(vPos),1.0);
    Color = vec4(1.0);
}