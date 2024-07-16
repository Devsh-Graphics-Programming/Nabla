#version 460 core

layout (location = 0) in vec4 VS_in_position;

layout (push_constant, row_major) uniform Block
{
	mat4 mvp;
} PushConstants;

layout(location = 0) out vec3 color;

void main()
{
	gl_Position = PushConstants.mvp * VS_in_position;
	color = vec3(1.f, 1.f, 0.f);
}