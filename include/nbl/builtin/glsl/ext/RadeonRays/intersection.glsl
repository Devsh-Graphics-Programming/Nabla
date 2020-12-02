#ifndef __NBL_EXT_RADEON_RAYS_INTERSECTION_INCLUDED__
#define __NBL_EXT_RADEON_RAYS_INTERSECTION_INCLUDED__

struct nbl_glsl_ext_RadeonRays_Intersection
{
	// Shape ID
	int shapeid;
	// Primitve ID
	int primid;

	int padding0;
	int padding1;

	// UV parametrization
	vec4 uvwt;
};

#endif