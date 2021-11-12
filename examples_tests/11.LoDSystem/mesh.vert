// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core

#include "common.glsl"

#define NBL_GLSL_TRANSFORM_TREE_POOL_DESCRIPTOR_SET 0
#define NBL_GLSL_TRANSFORM_TREE_POOL_NODE_GLOBAL_TRANSFORM_DESCRIPTOR_BINDING 0
// disable what we dont use
#define NBL_GLSL_TRANSFORM_TREE_POOL_NODE_PARENT_DESCRIPTOR_DECLARED
#define NBL_GLSL_TRANSFORM_TREE_POOL_NODE_RELATIVE_TRANSFORM_DESCRIPTOR_DECLARED
#define NBL_GLSL_TRANSFORM_TREE_POOL_NODE_MODIFIED_TIMESTAMP_DESCRIPTOR_DECLARED
#define NBL_GLSL_TRANSFORM_TREE_POOL_NODE_RECOMPUTED_TIMESTAMP_DESCRIPTOR_DECLARED
#include <nbl/builtin/glsl/transform_tree/pool_descriptor_set.glsl>

layout(set = 1, binding = 0, std430, row_major) restrict readonly buffer PerViewPerInstance
{
    PerViewPerInstance_t data[];
} perViewPerInstance;

layout(location = 0) in vec3 vPos;
layout(location = 3) in vec3 vNormal;
layout(location = 15) in uvec2 InstanceGUID_PerViewPerInstance;

layout(location = 0) out vec3 Normal;
layout(location = 1) out flat uint LoD;

#include <nbl/builtin/glsl/utils/transform.glsl>
void main()
{
	const PerViewPerInstance_t pvpi = perViewPerInstance.data[InstanceGUID_PerViewPerInstance[1]];
    const mat4x3 worldMatrix = nodeGlobalTransforms.data[InstanceGUID_PerViewPerInstance[0]];
	const mat3 fastNormalMatrix = nbl_glsl_sub3x3TransposeCofactors(mat3(worldMatrix));

    gl_Position = nbl_glsl_pseudoMul4x4with3x1(pvpi.mvp,vPos);
    // have to normalize both in vertex and fragment shader, because normal quantization foreshortens `vNormal` and `fastNormalMatrix` is missing a division by determinant
    Normal = normalize(fastNormalMatrix*vNormal);
    LoD = pvpi.lod;
}
