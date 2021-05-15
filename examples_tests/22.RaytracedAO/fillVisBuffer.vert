// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#version 430 core
#extension GL_EXT_shader_16bit_storage : require

#include "rasterizationCommon.h"

#define _NBL_VG_SSBO_DESCRIPTOR_SET 0
#include "virtualGeometry.glsl"

layout(set=1, binding=0, row_major) readonly restrict buffer PerInstancePerCamera
{
    DrawData_t data[];
} instanceDataPerCamera;


layout(location = 0) in vec3 vPosition;
layout(location = 2) in vec2 vUV;
layout(location = 3) in vec3 vNormal;

layout(location = 0) flat out uint BackfacingBit_ObjectID;
layout(location = 1) out vec3 Normal;
layout(location = 2) out vec2 UV;


#include <nbl/builtin/glsl/utils/transform.glsl>

void main()
{
    DrawData_t self = instanceDataPerCamera.data[gl_InstanceIndex];
    BackfacingBit_ObjectID = self.backfacingBit_objectID;

    gl_Position = nbl_glsl_pseudoMul4x4with3x1(self.MVP,vPosition);
    
    Normal = normalize(vNormal);
	
    UV = vUV;
}
