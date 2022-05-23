#ifndef _NBL_GLSL_BLIT_PARAMETERS_INCLUDED_
#define _NBL_GLSL_BLIT_PARAMETERS_INCLUDED_

#ifdef __cplusplus
#define uint uint32_t
#endif

struct nbl_glsl_blit_parameters_t
{
	uvec3 outDim;
	float referenceAlpha;
	vec3 fScale;
	uint inPixelCount;
	vec3 negativeSupport;
	uint outPixelCount;
	uvec3 windowDim;
	uint _pad0;
	uvec3 phaseCount;
	uint windowsPerWG;
};

#ifdef __cplusplus
#undef uint
#endif

#endif