
#ifndef _IRR_BUILTIN_GLSL_BUMP_MAPPING_UTILS_INCLUDED_
#define _IRR_BUILTIN_GLSL_BUMP_MAPPING_UTILS_INCLUDED_

vec3 irr_glsl_perturbNormal_heightMap(in vec3 vtxN, in mat2x3 dPdScreen, in vec2 dHdScreen)
{
    vec3 r1 = cross(dPdScreen[1], vtxN);
    vec3 r2 = cross(vtxN, dPdScreen[0]);
    vec3 surfGrad = (r1 * dHdScreen.x + r2 * dHdScreen.y) / dot(dPdScreen[0], r1);
    return normalize(vtxN - surfGrad);
}

#endif