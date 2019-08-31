#version 430 core
uniform mat4 MVP;

layout(location = 0) in vec4 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 2) in vec2 vTexCoord;

out vec2 TexCoord;

void main()
{
    gl_Position = MVP*vPos;
	TexCoord = vTexCoord;
}
