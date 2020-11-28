#ifndef __IRR_EXT_RADEON_RAYS_RAY_INCLUDED__
#define __IRR_EXT_RADEON_RAYS_RAY_INCLUDED__

struct irr_glsl_ext_RadeonRays_ray
{
	vec3 origin;
	float maxT; // FLT_MAX
	vec3 direction;
	float time;
	int mask; // want to have it to -1
	int _active; // want to have it to 1
	int backfaceCulling; // want to have it to 0
	int useless_padding; // can be used to forward data
};

irr_glsl_ext_RadeonRays_ray irr_glsl_ext_RadeonRays_constructDefaultRay(in vec3 origin, in vec3 direction, in float maxLen, in int userData)
{
	RadeonRays_ray retval;
	retval.origin = origin;
	retval.maxT = maxLen;
	retval.direction = direction;
	retval.time = 0.0;
	retval.mask = -1;
	retval._active = 1;
	retval.backfaceCulling = 0;
	retval.useless_padding = userData;
	return retval;
}

#endif