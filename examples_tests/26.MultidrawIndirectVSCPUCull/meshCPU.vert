#version 430 core

#include "commonVertexShader.glsl"

layout(push_constant) uniform PushConstants
{
    uint objectUUID;
} pc;

void main()
{
	impl(pc.objectUUID);
}
