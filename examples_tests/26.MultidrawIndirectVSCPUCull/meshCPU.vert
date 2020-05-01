#version 430 core

#include "../commonVertexShader.glsl"

layout(push_constant) PushConstants
{
    uint objectUUID;
};

void main()
{
	impl(objectUUID);
}
