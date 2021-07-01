// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#version 460 core
#extension GL_EXT_shader_16bit_storage : require
#include <nbl/builtin/glsl/barycentric/extensions.glsl>


#define _NBL_GLSL_EXT_MITSUBA_LOADER_INSTANCE_DATA_BINDING_ 0
#include "virtualGeometry.glsl"

#include <nbl/builtin/glsl/barycentric/frag.glsl>
layout(location = 2) flat in uint BackfacingBit_BatchInstanceGUID;
layout(location = 3) flat in uint drawCmdFirstIndex;

uint nbl_glsl_barycentric_frag_getDrawID() {return BackfacingBit_BatchInstanceGUID&0x7fffffffu;}
vec3 nbl_glsl_barycentric_frag_getVertexPos(in uint batchInstanceGUID, in uint primID, in uint primsVx)
{
    const uint ix = nbl_glsl_VG_fetchTriangleVertexIndex(primID*3u+drawCmdFirstIndex,primsVx);
    return nbl_glsl_fetchVtxPos(ix,InstData.data[batchInstanceGUID]);
}


layout(location = 0) out uvec4 frontFacingTriangleIDDrawID_unorm16Bary_dBarydScreenHalf2x2; // should it be called backfacing or frontfacing?


void main()
{
    vec2 bary = nbl_glsl_barycentric_frag_get();

    const int triangleIDBitcount = findMSB(MAX_TRIANGLES_IN_BATCH-1)+1;
	frontFacingTriangleIDDrawID_unorm16Bary_dBarydScreenHalf2x2[0] = bitfieldInsert(BackfacingBit_BatchInstanceGUID,gl_PrimitiveID,31-triangleIDBitcount,triangleIDBitcount)^(gl_FrontFacing ? 0x0u:0x80000000u);
    frontFacingTriangleIDDrawID_unorm16Bary_dBarydScreenHalf2x2[1] = packUnorm2x16(bary);
    frontFacingTriangleIDDrawID_unorm16Bary_dBarydScreenHalf2x2[2] = packHalf2x16(dFdx(bary));
    frontFacingTriangleIDDrawID_unorm16Bary_dBarydScreenHalf2x2[3] = packHalf2x16(dFdy(bary));
}
