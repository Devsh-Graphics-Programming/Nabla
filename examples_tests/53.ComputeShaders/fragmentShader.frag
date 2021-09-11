#version 430 core

layout(location = 0) in vec4 inFFullyProjectedVelocity;
layout(location = 1) in vec4 inFColor;

layout(location = 0) out vec4 outColor;

void main()
{
    outColor = inFColor;
}
		