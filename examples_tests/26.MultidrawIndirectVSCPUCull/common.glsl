struct ModelData_t
{
#ifdef __cplusplus
    core::matrix3x4SIMD   worldMatrix;
    core::matrix3x4SIMD normalMatrix;
    core::vectorSIMDf   bbox[2];
#else
    mat4x3  worldMatrix;
    mat3    normalMatrix;
    vec3    bbox[2];
#endif
};

struct DrawData_t
{
#ifdef __cplusplus
    core::matrix4SIMD   modelViewProjMatrix;
    core::matrix3x4SIMD normalMatrix;
#else
    mat4 modelViewProjMatrix;
    mat3 normalMatrix;
#endif
};

#ifndef __cplusplus
#include <irr/builtin/glsl/vertex_utils/vertex_utils.glsl>

// TODO: someone add this to "irr/builtin/glsl/vertex_utils.glsl" later
mat4x3 irr_glsl_pseudoMul4x3with4x3(in mat4x3 lhs, in mat4x3 rhs)
{
    mat4x3 result;
    for (int i=0; i<4; i++)
        result[i] = lhs[0]*rhs[i][0]+lhs[1]*rhs[i][1]+lhs[2]*rhs[i][2];
    result[3] += lhs[3];
    return result;
}

mat4 irr_glsl_pseudoMul4x4with4x3(in mat4 proj, in mat4x3 tform)
{
    mat4 result;
    for (int i=0; i<4; i++)
        result[i] = proj[0]*tform[i][0]+proj[1]*tform[i][1]+proj[2]*tform[i][2];
    result[3] += proj[3];
    return result;
}

bool irr_glsl_couldBeVisible(in mat4 proj, in mat2x3 bbox)
{
    mat4 pTpose = transpose(proj);
    mat4 xyPlanes = mat4(pTpose[3]+pTpose[0],pTpose[3]+pTpose[1],pTpose[3]-pTpose[0],pTpose[3]-pTpose[1]);
    vec4 farPlane = pTpose[3]+pTpose[2];

#define getClosestDP(R) (dot(mix(bbox[1],bbox[0],lessThan(R.xyz,vec3(0.f)) ),R.xyz)+R.w>0.f)

    return  getClosestDP(xyPlanes[0])&&getClosestDP(xyPlanes[1])&&
            getClosestDP(xyPlanes[2])&&getClosestDP(xyPlanes[3])&&
            getClosestDP(pTpose[3])&&getClosestDP(farPlane);

#undef getClosestDP
}
#endif