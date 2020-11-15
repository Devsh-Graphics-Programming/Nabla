#version 450 core

#include "shaderCommon.glsl"

layout(location = 0) in vec4 gFullyProjectedVelocity[];
layout(location = 1) in vec4 gColor[];

layout(location = 0) out vec4 outFVelocity;
layout(location = 1) out vec4 outFColor;

layout (points) in;
layout (line_strip, max_vertices = 2) out;

void main()
{
	if(pushConstants.isCPressed)
	{
		outFColor = vec4(0.0, 1.0, 0.0, 0.0);
		gl_Position = gl_in[0].gl_Position;
		EmitVertex();
		gl_Position = gl_in[0].gl_Position + gFullyProjectedVelocity[0];
		EmitVertex();

		EndPrimitive();
	}
}
		