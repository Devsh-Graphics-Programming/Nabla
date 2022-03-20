#ifndef _NBL_GLSL_BLIT_PARAMETERS_INCLUDED_
#define _NBL_GLSL_BLIT_PARAMETERS_INCLUDED_

struct nbl_glsl_blit_parameters_t
{
	uvec3 inDim;
	uvec3 outDim;

	vec3 negativeSupport;
	vec3 positiveSupport;

	uvec3 windowDim;
	uvec3 phaseCount;

	uint padding;
	uint windowsPerWG;
	uint axisCount;
};

#endif