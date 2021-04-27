#version 430 core

layout(location = 0) flat in uint drawID;

// TODO: investigate using snorm16 for the derivatives
layout(location = 0) out uvec4 triangleIDdrawID_unorm16Bary_dBarydScreenHalf2x2; // 32bit triangleID, 2x16bit barycentrics, 4x16bit barycentric derivatives 

#include "common.glsl"

//TODO: barycentric shit include
void main()
{
    vec2 bary = vec2(0.0,0.0);

    triangleIDdrawID_unorm16Bary_dBarydScreenHalf2x2[0] = bitfieldInsert(gl_PrimitiveID,drawID,MAX_TRIANGLES_IN_BATCH,32-MAX_TRIANGLES_IN_BATCH);
    triangleIDdrawID_unorm16Bary_dBarydScreenHalf2x2[1] = packUnorm2x16(bary);
    triangleIDdrawID_unorm16Bary_dBarydScreenHalf2x2[2] = packHalf2x16(dFdx(bary));
    triangleIDdrawID_unorm16Bary_dBarydScreenHalf2x2[3] = packHalf2x16(dFdy(bary));
}