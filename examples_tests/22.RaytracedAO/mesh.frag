#version 430 core

#include "irr/builtin/glsl/utils/normal_encode.glsl"


layout(location = 0) flat in uint ObjectID;
layout(location = 1) in vec3 Normal;
layout(location = 2) in vec2 UV;

layout(location = 0) out uvec2 objectTriangleFrontFacing;
layout(location = 1) out vec2 encodedNormal;
layout(location = 2) out vec2 uv;

void main()
{		
	objectTriangleFrontFacing = uvec2(ObjectID^(gl_FrontFacing ? 0x0u:0x80000000u),gl_PrimitiveID);
	encodedNormal = irr_glsl_NormalEncode_signedSpherical(normalize(Normal));
	uv = UV;
}
