#version 430 core
#extension GL_ARB_shader_draw_parameters : require

layout(location = 0) uniform mat4 MVP;

layout(location = 0) in vec4 vPos;
layout(location = 2) in vec2 vTexCoord;


layout(location = 0) out flat uint DrawID;
layout(location = 1) out vec2 TexCoord;

void main()
{
    gl_Position = MVP*vPos;
	DrawID = gl_DrawIDARB;
	TexCoord = vTexCoord;
}
