#version 460 core
#include <nbl/builtin/glsl/barycentric/extensions.glsl>

#include "common.glsl"

layout(location = 0) flat in uint drawGUID;

// TODO: investigate using snorm16 for the derivatives
layout(location = 0) out uvec4 triangleIDdrawID_unorm16Bary_dBarydScreenHalf2x2; // 32bit triangleID, 2x16bit barycentrics, 4x16bit barycentric derivatives 
#include <nbl/builtin/glsl/barycentric/frag.glsl>

uint nbl_glsl_barycentric_frag_getDrawID() {return drawGUID;}
vec3 nbl_glsl_barycentric_frag_getVertexPos(in uint drawID, in uint primID, in uint primsVx)
{
    const uint ix = nbl_glsl_VG_fetchTriangleVertexIndex(primID*3u+batchInstanceData[drawID].firstIndex,primsVx);
    return nbl_glsl_fetchVtxPos(ix,drawID);
}

void main()
{
    vec2 bary = nbl_glsl_barycentric_frag_get();

    const int triangleIDBitcount = findMSB(MAX_TRIANGLES_IN_BATCH-1)+1;
    triangleIDdrawID_unorm16Bary_dBarydScreenHalf2x2[0] = bitfieldInsert(gl_PrimitiveID,drawGUID,triangleIDBitcount,32-triangleIDBitcount);
    triangleIDdrawID_unorm16Bary_dBarydScreenHalf2x2[1] = packUnorm2x16(bary);
    triangleIDdrawID_unorm16Bary_dBarydScreenHalf2x2[2] = packHalf2x16(dFdx(bary));
    triangleIDdrawID_unorm16Bary_dBarydScreenHalf2x2[3] = packHalf2x16(dFdy(bary));
}