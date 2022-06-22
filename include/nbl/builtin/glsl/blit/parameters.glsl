#ifndef _NBL_GLSL_BLIT_PARAMETERS_INCLUDED_
#define _NBL_GLSL_BLIT_PARAMETERS_INCLUDED_

#ifdef __cplusplus
#define uint uint32_t
#endif

struct nbl_glsl_blit_parameters_t
{
	uvec3 inDim;
	//! Offset into the shared memory array which tells us from where the second buffer of shared memory begins
	//! Given by max(memory_for_preload_region, memory_for_result_of_y_pass)
	uint secondScratchOffset; 
	uvec3 outDim;
	float referenceAlpha;
	vec3 fScale;
	uint inPixelCount;
	vec3 negativeSupport;
	uint outPixelCount;
	uvec3 windowDim;
	uint _pad0;
	uvec3 phaseCount;
	uint _pad1; // if we end up removing windowDim, we can add a padding here
	uvec3 outputTexelsPerWG;
	uint _pad2;
	uvec3 preloadRegion;
};

#ifdef __cplusplus
#undef uint
#endif

#endif