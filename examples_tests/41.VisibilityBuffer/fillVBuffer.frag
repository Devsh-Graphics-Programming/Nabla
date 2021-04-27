#version 430 core

layout(location = 0) flat in uint drawID;

// TODO: investigate using snorm16 for the derivatives
layout(location = 0) out uvec4 triangleIDdrawID_unorm16Bary_dBarydScreenHalf2x2; // 32bit triangleID, 2x16bit barycentrics, 4x16bit barycentric derivatives 

#include "common.glsl"

//TODO: move this to some barycentric header
vec2 nbl_glsl_barycentric_reconstructBarycentrics(in vec3 positionRelativeToV0, in mat2x3 edges)
{
    const float e0_2 = dot(edges[0],edges[0]);
    const float e0e1 = dot(edges[0],edges[1]);
    const float e1_2 = dot(edges[1],edges[1]);

    const float qe0 = dot(positionRelativeToV0,edges[0]);
    const float qe1 = dot(positionRelativeToV0,edges[1]);
    vec2 protoBary = vec2(qe0*e1_2-qe1*e0e1,qe1*e0_2-qe0*e0e1);

    const float rcp_dep = 1.f/(e0_2*e1_2+e0e1*e0e1);
    return protoBary*rcp_dep;
}
vec2 nbl_glsl_barycentric_reconstructBarycentrics(in vec3 pointPosition, in mat3 vertexPositions)
{
    return nbl_glsl_barycentric_reconstructBarycentrics(pointPosition-vertexPositions[0],mat2x3(vertexPositions[1]-vertexPositions[0],vertexPositions[2]-vertexPositions[0]));
}



// TODO: move this to vertex shader barycentric header
void nbl_glsl_barycentric_vert_set(in vec3 pos);

// TODO: Check for Nvidia Pascal Barycentric extension
#if 0
// TODO: Check for AMD Barycentric extension
#elif 0
//
#else
#define NBL_GLSL_BARYCENTRIC_VERT_POS_INPUT_LOC 0
#define NBL_GLSL_BARYCENTRIC_VERT_PROVOKINGPOS_INPUT_LOC 1
layout(location = NBL_GLSL_BARYCENTRIC_VERT_POS_INPUT_LOC) out vec3 nbl_glsl_barycentric_vert_pos;
layout(location = NBL_GLSL_BARYCENTRIC_VERT_PROVOKINGPOS_INPUT_LOC) flat out vec3 nbl_glsl_barycentric_vert_provokingPos;

void nbl_glsl_barycentric_vert_set(in vec3 pos)
{
    nbl_glsl_barycentric_vert_pos = pos;
    nbl_glsl_barycentric_vert_provokingPos = pos;
}
#endif


// TODO: move this to fragment shader barycentric header

vec2 nbl_glsl_barycentric_frag_get();

// TODO: Check for Nvidia Pascal Barycentric extension
#if 0
// TODO: Check for AMD Barycentric extension
#elif 0
//
#else
#define NBL_GLSL_BARYCENTRIC_FRAG_POS_INPUT_LOC 0
#define NBL_GLSL_BARYCENTRIC_FRAG_PROVOKINGPOS_INPUT_LOC 1
layout(location = NBL_GLSL_BARYCENTRIC_FRAG_POS_INPUT_LOC) in vec3 nbl_glsl_barycentric_frag_pos;
layout(location = NBL_GLSL_BARYCENTRIC_FRAG_PROVOKINGPOS_INPUT_LOC) flat in vec3 nbl_glsl_barycentric_frag_provokingPos;

uint nbl_glsl_barycentric_frag_getDrawID();
vec3 nbl_glsl_barycentric_frag_getVertexPos(in uint drawID, in uint primID, in uint primsVx);

vec2 nbl_glsl_barycentric_frag_get()
{
    return nbl_glsl_barycentric_reconstructBarycentrics(
        nbl_glsl_barycentric_frag_pos-nbl_glsl_barycentric_frag_provokingPos,
        mat2x3(
            nbl_glsl_barycentric_frag_getVertexPos(nbl_glsl_barycentric_frag_getDrawID(),gl_PrimitiveID,1u)-nbl_glsl_barycentric_frag_provokingPos,
            nbl_glsl_barycentric_frag_getVertexPos(nbl_glsl_barycentric_frag_getDrawID(),gl_PrimitiveID,2u)-nbl_glsl_barycentric_frag_provokingPos
        )
    );
}
#endif

void main()
{
    vec2 bary = nbl_glsl_barycentric_frag_get();

    triangleIDdrawID_unorm16Bary_dBarydScreenHalf2x2[0] = bitfieldInsert(gl_PrimitiveID,drawID,MAX_TRIANGLES_IN_BATCH,32-MAX_TRIANGLES_IN_BATCH);
    triangleIDdrawID_unorm16Bary_dBarydScreenHalf2x2[1] = packUnorm2x16(bary);
    triangleIDdrawID_unorm16Bary_dBarydScreenHalf2x2[2] = packHalf2x16(dFdx(bary));
    triangleIDdrawID_unorm16Bary_dBarydScreenHalf2x2[3] = packHalf2x16(dFdy(bary));
}