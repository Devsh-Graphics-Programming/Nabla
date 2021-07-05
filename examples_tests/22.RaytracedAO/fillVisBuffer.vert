// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#version 460 core
#extension GL_EXT_shader_16bit_storage : require
#include <nbl/builtin/glsl/barycentric/extensions.glsl>

#include "rasterizationCommon.h"

#define _NBL_GLSL_EXT_MITSUBA_LOADER_INSTANCE_DATA_BINDING_ 0
#include "virtualGeometry.glsl"

layout(set=2, binding=0, row_major) readonly restrict buffer PerInstancePerCamera
{
    DrawData_t data[];
} instanceDataPerCamera;

#include <nbl/builtin/glsl/barycentric/vert.glsl>
layout(location = 2) flat out uint BackfacingBit_BatchInstanceGUID;
layout(location = 3) flat out uint drawCmdFirstIndex;

#include <nbl/builtin/glsl/utils/transform.glsl>
void main()
{
    DrawData_t self = instanceDataPerCamera.data[gl_InstanceIndex];
    BackfacingBit_BatchInstanceGUID = self.backfacingBit_batchInstanceGUID;
    drawCmdFirstIndex = self.firstIndex;

    const uint batchInstanceGUID = self.backfacingBit_batchInstanceGUID&0x7fffffffu;
    
    const vec3 modelPos = nbl_glsl_fetchVtxPos(gl_VertexIndex,InstData.data[batchInstanceGUID]);
    nbl_glsl_barycentric_vert_set(modelPos);
    gl_Position = nbl_glsl_pseudoMul4x4with3x1(self.MVP,modelPos);
}
