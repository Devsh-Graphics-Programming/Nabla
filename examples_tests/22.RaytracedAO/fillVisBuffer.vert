// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#version 430 core
#extension GL_EXT_shader_16bit_storage : require

#include "rasterizationCommon.h"

#define _NBL_GLSL_EXT_MITSUBA_LOADER_INSTANCE_DATA_BINDING_ 0
#include "virtualGeometry.glsl"

layout(set=2, binding=0, row_major) readonly restrict buffer PerInstancePerCamera
{
    DrawData_t data[];
} instanceDataPerCamera;


layout(location = 0) flat out uint BackfacingBit_BatchInstanceGUID;
layout(location = 1) out vec3 Normal;
layout(location = 2) out vec2 UV;


#include <nbl/builtin/glsl/utils/transform.glsl>

void main()
{
    DrawData_t self = instanceDataPerCamera.data[gl_InstanceIndex];
    BackfacingBit_BatchInstanceGUID = self.backfacingBit_batchInstanceGUID;

    const uint batchInstanceGUID = self.backfacingBit_batchInstanceGUID&0x0fffffffu;

    gl_Position = nbl_glsl_pseudoMul4x4with3x1(self.MVP,nbl_glsl_fetchVtxPos(gl_VertexIndex,batchInstanceGUID));
    
    Normal = normalize(nbl_glsl_fetchVtxNormal(gl_VertexIndex,batchInstanceGUID));
	
    UV = nbl_glsl_fetchVtxUV(gl_VertexIndex,batchInstanceGUID);
}
