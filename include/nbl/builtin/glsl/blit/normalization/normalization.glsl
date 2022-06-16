#ifndef _NBL_GLSL_BLIT_NORMALIZATION_INCLUDED_
#define _NBL_GLSL_BLIT_NORMALIZATION_INCLUDED_

#include <nbl/builtin/glsl/workgroup/arithmetic.glsl>

//! WARNING: ONLY WORKS FOR `dividendMsb<=2^23` DUE TO FP32 ABUSE !!!
uint integerDivide_64_32_32(in uint dividendMsb, in uint dividendLsb, in uint divisor)
{
	//assert(dividendMsb<=(0x1u<<23));
	const uint MAX_UINT = ~0u;
	const float msbRatio = float(dividendMsb) / float(divisor);
	const uint quotient = uint((msbRatio * MAX_UINT) + msbRatio) + dividendLsb / divisor;
	return quotient;
}

#ifndef _NBL_GLSL_BLIT_NORMALIZATION_MAIN_DEFINED_

#ifndef _NBL_GLSL_SCRATCH_SHARED_DEFINED_
#error "_NBL_GLSL_SCRATCH_SHARED_DEFINED_ must be defined"
#endif

#define scratchShared _NBL_GLSL_SCRATCH_SHARED_DEFINED_

#include <nbl/builtin/glsl/blit/parameters.glsl>
nbl_glsl_blit_parameters_t nbl_glsl_blit_getParameters();

vec4 nbl_glsl_blit_normalization_getData(in uvec3 coord, in uint layerIdx);
void nbl_glsl_blit_normalization_setData(in vec4 data, in uvec3 coord, in uint layerIdx);
uint nbl_glsl_blit_normalization_getAlphaHistogramData(in uint index, in uint layerIdx);
uint nbl_glsl_blit_normalization_getPassedInputPixelCount(in uint layerIdx);

void nbl_glsl_blit_normalization_main()
{
	uint histogramVal = 0u;
	if (gl_LocalInvocationIndex < _NBL_GLSL_BLIT_ALPHA_BIN_COUNT_)
		histogramVal = nbl_glsl_blit_normalization_getAlphaHistogramData(gl_LocalInvocationIndex, gl_WorkGroupID.z);

	const uint cumHistogramVal = nbl_glsl_workgroupInclusiveAdd(histogramVal);

	if (gl_LocalInvocationIndex < _NBL_GLSL_BLIT_ALPHA_BIN_COUNT_)
		scratchShared[gl_LocalInvocationIndex] = cumHistogramVal;
	barrier();

	const nbl_glsl_blit_parameters_t params = nbl_glsl_blit_getParameters();

	const uint outputPixelCount = params.outPixelCount;

	uint productMsb, productLsb;
	umulExtended(nbl_glsl_blit_normalization_getPassedInputPixelCount(gl_WorkGroupID.z), outputPixelCount, productMsb, productLsb);

	const uint pixelsShouldPassCount = integerDivide_64_32_32(productMsb, productLsb, params.inPixelCount);
	const uint pixelsShouldFailCount = outputPixelCount - pixelsShouldPassCount;

	if ((pixelsShouldFailCount <= cumHistogramVal) && ((gl_LocalInvocationIndex == 0) || (scratchShared[gl_LocalInvocationIndex - 1] < pixelsShouldFailCount)))
	{
		const uint bucketIndex = gl_LocalInvocationIndex;
		const float newReferenceAlpha = min((bucketIndex - 0.5f) / float(_NBL_GLSL_BLIT_ALPHA_BIN_COUNT_ - 1), 1.f);

		scratchShared[0] = floatBitsToUint(params.referenceAlpha / newReferenceAlpha);
	}
	barrier();

	const float alphaScale = uintBitsToFloat(scratchShared[0]);

	if (all(lessThan(gl_GlobalInvocationID, params.outDim)))
	{
		const vec4 pixel = nbl_glsl_blit_normalization_getData(gl_GlobalInvocationID, gl_WorkGroupID.z);
		const vec4 scaledPixel = vec4(pixel.rgb, pixel.a * alphaScale);
		nbl_glsl_blit_normalization_setData(scaledPixel, gl_GlobalInvocationID, gl_WorkGroupID.z);
	}
}

#undef scratchShared

#define _NBL_GLSL_BLIT_NORMALIZATION_MAIN_DEFINED_
#endif

#endif
