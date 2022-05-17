#ifndef _NBL_GLSL_BLIT_NORMALIZATION_INCLUDED_
#define _NBL_GLSL_BLIT_NORMALIZATION_INCLUDED_


#include <nbl/builtin/glsl/workgroup/arithmetic.glsl>

uint integerDivide_64_32_32(in uint dividendMsb, in uint dividendLsb, in uint divisor)
{
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

#include <nbl/builtin/glsl/blit/normalization/parameters.glsl>
nbl_glsl_blit_normalization_parameters_t nbl_glsl_blit_normalization_getParameters();

nbl_glsl_blit_normalization_pixel_t nbl_glsl_blit_normalization_getPaddedData(in ivec3 coord);
void nbl_glsl_blit_normalization_setData(in nbl_glsl_blit_normalization_pixel_t data, in ivec3 coord);
uint nbl_glsl_blit_normalization_getAlphaHistogramData(in uint index);
uint nbl_glsl_blit_normalization_getPassedInputPixelCountData();

// #include <nbl/builtin/glsl/algorithm.glsl>
// NBL_GLSL_DEFINE_LOWER_BOUND(scratchShared, uint);

void nbl_glsl_blit_normalization_main()
{
	uint histogramVal = 0u;
	if (gl_LocalInvocationIndex < _NBL_GLSL_BLIT_NORMALIZATION_BIN_COUNT_)
		histogramVal = nbl_glsl_blit_normalization_getAlphaHistogramData(gl_LocalInvocationIndex);

	const uint cumHistogramVal = nbl_glsl_workgroupInclusiveAdd(histogramVal);

	if (gl_LocalInvocationIndex < _NBL_GLSL_BLIT_NORMALIZATION_BIN_COUNT_)
		scratchShared[gl_LocalInvocationIndex] = cumHistogramVal;
	barrier();

	const nbl_glsl_blit_normalization_parameters_t params = nbl_glsl_blit_normalization_getParameters();

	const uint outputPixelCount = params.outImageDim.x * params.outImageDim.y * params.outImageDim.z;

	uint productMsb, productLsb;
	umulExtended(nbl_glsl_blit_normalization_getPassedInputPixelCountData(), outputPixelCount, productMsb, productLsb);

	const uint pixelsShouldPassCount = integerDivide_64_32_32(productMsb, productLsb, params.inPixelCount);
	const uint pixelsShouldFailCount = outputPixelCount - pixelsShouldPassCount;

	uint bucketIndex;
	{
		uint begin = 0u;
		const uint end = _NBL_GLSL_BLIT_NORMALIZATION_BIN_COUNT_;
		const uint value = pixelsShouldFailCount;
		uint len = end - begin;
		if (NBL_GLSL_IS_NOT_POT(len))
		{
			const uint newLen = 0x1u << findMSB(len);
			const uint diff = len - newLen;

			begin = NBL_GLSL_LESS(value, NBL_GLSL_EVAL(scratchShared)[newLen]) ? 0u : diff;
			len = newLen;
		}

		while (len != 0u)
		{
			begin += NBL_GLSL_LESS(value, NBL_GLSL_EVAL(scratchShared)[begin + (len >>= 1u)]) ? 0u : len;
			begin += NBL_GLSL_LESS(value, NBL_GLSL_EVAL(scratchShared)[begin + (len >>= 1u)]) ? 0u : len;
		}

		bucketIndex = begin + (NBL_GLSL_LESS(value, NBL_GLSL_EVAL(scratchShared)[begin]) ? 0u : 1u);
	}

	// const uint bucketIndex = lower_bound_scratchShared_NBL_GLSL_LESS(0u, _NBL_GLSL_BLIT_NORMALIZATION_BIN_COUNT_, pixelsShouldFailCount);

	const float newReferenceAlpha = min((bucketIndex - 0.5f) / float(_NBL_GLSL_BLIT_NORMALIZATION_BIN_COUNT_ - 1), 1.f);

	const float alphaScale = params.oldReferenceAlpha / newReferenceAlpha;

	const nbl_glsl_blit_normalization_pixel_t pixel = nbl_glsl_blit_normalization_getPaddedData(ivec3(gl_GlobalInvocationID));
	nbl_glsl_blit_normalization_pixel_t scaledPixel;
	scaledPixel.data = vec4(pixel.data.rgb, pixel.data.a * alphaScale);
	nbl_glsl_blit_normalization_setData(scaledPixel, ivec3(gl_GlobalInvocationID));
}

#undef scratchShared

#define _NBL_GLSL_BLIT_NORMALIZATION_MAIN_DEFINED_
#endif

#endif