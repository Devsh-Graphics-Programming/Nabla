#ifndef _NBL_BUILTIN_GLSL_SAMPLING_ENVMAP_INCLUDED_
#define _NBL_BUILTIN_GLSL_SAMPLING_ENVMAP_INCLUDED_

#include <nbl/builtin/glsl/math/constants.glsl>

vec2 nbl_glsl_sampling_envmap_uvCoordFromDirection(vec3 v)
{
    vec2 uv = vec2(atan(v.y,v.x),acos(v.z));
    uv.x *= nbl_glsl_RECIPROCAL_PI*0.5;
	if (v.y<0.f)
		uv.x += 1.f;
    uv.y *= nbl_glsl_RECIPROCAL_PI;
    return uv;
}

vec3 nbl_glsl_sampling_envmap_directionFromUVCoord(in vec2 uv, out float sinTheta)
{
	vec3 dir;
	dir.x = cos(uv.x*2.f*nbl_glsl_PI);
	dir.y = sqrt(1.f-dir.x*dir.x);
	if (uv.x>0.5f)
		dir.y = -dir.y;
	nbl_glsl_sincos(uv.y*nbl_glsl_PI,sinTheta,dir.z);
	dir.xy *= sinTheta;
	return dir;
}

#endif
