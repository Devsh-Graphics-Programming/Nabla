#ifndef _IRR_BUILTIN_GLSL_UTILS_CULLING_INCLUDED_
#define _IRR_BUILTIN_GLSL_UTILS_CULLING_INCLUDED_

// TODO: culling include
bool irr_glsl_couldBeVisible(in mat4 proj, in mat2x3 bbox)
{
    mat4 pTpose = transpose(proj);
    mat4 xyPlanes = mat4(pTpose[3] + pTpose[0], pTpose[3] + pTpose[1], pTpose[3] - pTpose[0], pTpose[3] - pTpose[1]);
    vec4 farPlane = pTpose[3] + pTpose[2];

#define getClosestDP(R) (dot(mix(bbox[1],bbox[0],lessThan(R.xyz,vec3(0.f)) ),R.xyz)+R.w>0.f)

    return  getClosestDP(xyPlanes[0]) && getClosestDP(xyPlanes[1]) &&
        getClosestDP(xyPlanes[2]) && getClosestDP(xyPlanes[3]) &&
        getClosestDP(pTpose[3]) && getClosestDP(farPlane);
#undef getClosestDP
}

#endif