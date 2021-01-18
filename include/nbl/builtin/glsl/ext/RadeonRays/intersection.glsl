#ifndef _NBL_EXT_RADEON_RAYS_INTERSECTION_INCLUDED_
#define _NBL_EXT_RADEON_RAYS_INTERSECTION_INCLUDED_

// for the love of god, lets optimize this into 16 bytes
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