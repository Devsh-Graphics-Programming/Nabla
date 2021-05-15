// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#version 430 core
#extension GL_EXT_shader_16bit_storage : require

#define _NBL_VG_SSBO_DESCRIPTOR_SET 0
#include "virtualGeometry.glsl"

#include <nbl/builtin/glsl/utils/normal_encode.glsl>

layout(location = 0) flat in uint BackfacingBit_ObjectID;
layout(location = 1) in vec3 Normal;
layout(location = 2) in vec2 UV;

layout(location = 0) out uvec2 frontFacing_Object_Triangle; // should it be called backfacing or frontfacing?
layout(location = 1) out vec2 encodedNormal;
layout(location = 2) out vec2 uv;

void main()
{		
	frontFacing_Object_Triangle = uvec2(BackfacingBit_ObjectID^(gl_FrontFacing ? 0x0u:0x80000000u),gl_PrimitiveID);
	// TODO: these will disappear once we finally have MeshPackerV2 and settle on a way to obtain barycentrics
	encodedNormal = nbl_glsl_NormalEncode_signedSpherical(normalize(Normal));
	uv = UV;
}
