#ifndef _NBL_EXT_RADEON_RAYS_INTERSECTION_INCLUDED_
#define _NBL_EXT_RADEON_RAYS_INTERSECTION_INCLUDED_

struct nbl_glsl_ext_RadeonRays_Intersection
{
	// Shape ID
	int shapeid;
	// Primitve ID
	int primid;
	// UV parametrization
	vec2 uv;
};

#endif