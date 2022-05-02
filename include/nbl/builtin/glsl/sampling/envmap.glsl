#ifndef _NBL_BUILTIN_GLSL_SAMPLING_ENVMAP_INCLUDED_
#define _NBL_BUILTIN_GLSL_SAMPLING_ENVMAP_INCLUDED_

#include <nbl/builtin/glsl/math/constants.glsl>

vec2 nbl_glsl_sampling_envmap_generateUVCoordFromDirection(vec3 v)
{
    vec2 uv = vec2(atan(v.z, v.x), acos(v.y));
    uv.x *= nbl_glsl_RECIPROCAL_PI*0.5;
    uv.x += 0.5; 
    uv.y *= nbl_glsl_RECIPROCAL_PI;
    return uv;
}

vec3 nbl_glsl_sampling_envmap_generateDirectionFromUVCoord(in vec2 uv, out float sinTheta)
{
	vec3 dir;
	nbl_glsl_sincos((uv.x-0.5)*2.f*nbl_glsl_PI,dir.y,dir.x);
	nbl_glsl_sincos(uv.y*nbl_glsl_PI,sinTheta,dir.z);
	dir.xy *= sinTheta;
	return dir;
}

#endif
