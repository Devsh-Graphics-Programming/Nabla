#version 430 core

// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

layout(location = 0) in vec3 vPos;
layout(location = 3) in vec3 vNormal;
layout(location = 15) in uint iObjectID;

layout(set=0, binding=0) readonly buffer InstanceColors
{
	vec4 data[];
} instanceColors;
layout(set=0, binding=1, row_major) readonly buffer InstanceTransforms
{
	mat4x3 data[];
} instanceTransforms;

layout( push_constant, row_major ) uniform Block
{
	mat4 viewProj;
} PushConstants;

layout(location = 0) out vec3 Color;
layout(location = 1) out vec3 Normal;

#include "nbl/builtin/glsl/utils/transform.glsl"

void main()
{
	const mat4x3 worldMat = instanceTransforms.data[iObjectID];

	gl_Position = nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,nbl_glsl_pseudoMul3x4with3x1(worldMat,vPos));
	Color = instanceColors.data[iObjectID].xyz;

	mat3 inverseTransposeWorld = inverse(transpose(mat3(worldMat)));
	Normal = inverseTransposeWorld * normalize(vNormal);
}