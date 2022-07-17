#ifndef _NBL_GLSL_BLIT_PARAMETERS_INCLUDED_
#define _NBL_GLSL_BLIT_PARAMETERS_INCLUDED_

#ifdef __cplusplus
#define uint uint32_t
#endif

struct nbl_glsl_blit_parameters_t
{
	uvec3 dims; // input dimensions in lower 16 bits, output dimensions in higher 16 bits
	//! Offset into the shared memory array which tells us from where the second buffer of shared memory begins
	//! Given by max(memory_for_preload_region, memory_for_result_of_y_pass)
	uint secondScratchOffset; 
	uvec3 iterationRegionXPrefixProducts;
	float referenceAlpha;
	vec3 fScale;
	uint inPixelCount;
	vec3 negativeSupport;
	uint outPixelCount;
	uvec3 windowDimPhaseCount; // windowDim in lower 16 bits, phaseCount in higher 16 bits
	uint kernelWeightsOffsetY;
	uvec3 iterationRegionYPrefixProducts;
	uint kernelWeightsOffsetZ;
	uvec3 iterationRegionZPrefixProducts;
	uint outputTexelsPerWGZ;
	uvec3 preloadRegion;
};

#ifndef __cplusplus

#ifndef _NBL_GLSL_BLIT_PARAMETERS_METHODS_DEFINED_
#define _NBL_GLSL_BLIT_PARAMETERS_METHODS_DEFINED_

nbl_glsl_blit_parameters_t nbl_glsl_blit_getParameters();

uvec3 nbl_glsl_blit_parameters_getInputImageDimensions()
{
	const nbl_glsl_blit_parameters_t params = nbl_glsl_blit_getParameters();
	return uvec3(bitfieldExtract(params.dims.x, 0, 16), bitfieldExtract(params.dims.y, 0, 16), bitfieldExtract(params.dims.z, 0, 16));
}

uvec3 nbl_glsl_blit_parameters_getOutputImageDimensions()
{
	const nbl_glsl_blit_parameters_t params = nbl_glsl_blit_getParameters();
	return uvec3(bitfieldExtract(params.dims.x, 16, 32), bitfieldExtract(params.dims.y, 16, 32), bitfieldExtract(params.dims.z, 16, 32));
}

uvec3 nbl_glsl_blit_parameters_getWindowDimensions()
{
	const nbl_glsl_blit_parameters_t params = nbl_glsl_blit_getParameters();
	return uvec3(bitfieldExtract(params.windowDimPhaseCount.x, 0, 16), bitfieldExtract(params.windowDimPhaseCount.y, 0, 16), bitfieldExtract(params.windowDimPhaseCount.z, 0, 16));
}

uvec3 nbl_glsl_blit_parameters_getPhaseCount()
{
	const nbl_glsl_blit_parameters_t params = nbl_glsl_blit_getParameters();
	return uvec3(bitfieldExtract(params.windowDimPhaseCount.x, 16, 32), bitfieldExtract(params.windowDimPhaseCount.y, 16, 32), bitfieldExtract(params.windowDimPhaseCount.z, 16, 32));
}

uvec3 nbl_glsl_blit_parameters_getOutputTexelsPerWG()
{
	//! `outputTexelsPerWG.xy` just happens to be in the first components of `iterationRegionsXPrefixProducts` and `iterationRegionYPrefixProducts` --this is
	//! the result of how we choose to iterate, i.e. if, in the future, we decide to iterate differently, this needs to change.
	const nbl_glsl_blit_parameters_t params = nbl_glsl_blit_getParameters();
	return uvec3(params.iterationRegionXPrefixProducts.x, params.iterationRegionYPrefixProducts.x, params.outputTexelsPerWGZ);

}

#endif

#endif

#ifdef __cplusplus
#undef uint
#endif

#endif