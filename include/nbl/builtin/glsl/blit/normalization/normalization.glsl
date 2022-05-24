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

vec4 nbl_glsl_blit_normalization_getPaddedData(in ivec3 coord);
void nbl_glsl_blit_normalization_setData(in vec4 data, in ivec3 coord);
uint nbl_glsl_blit_normalization_getAlphaHistogramData(in uint index, in uint layerIdx);
uint nbl_glsl_blit_normalization_getPassedInputPixelCountData(in uint layerIdx);

// #include <nbl/builtin/glsl/algorithm.glsl>
// NBL_GLSL_DEFINE_LOWER_BOUND(scratchShared, uint);

void nbl_glsl_blit_normalization_main()
{
	// Todo(achal): Need to pull this out
#if NBL_GLSL_EQUAL(_NBL_GLSL_BLIT_DIM_COUNT_, 1)
	#define LAYER_IDX gl_GlobalInvocationID.y
#elif NBL_GLSL_EQUAL(_NBL_GLSL_BLIT_DIM_COUNT_, 2)
	#define LAYER_IDX gl_GlobalInvocationID.z
#elif NBL_GLSL_EQUAL(_NBL_GLSL_BLIT_DIM_COUNT_, 3)
	#define LAYER_IDX 0
#else
	#error _NBL_GLSL_BLIT_DIM_COUNT_ not supported
#endif

	uint histogramVal = 0u;
	if (gl_LocalInvocationIndex < _NBL_GLSL_BLIT_ALPHA_BIN_COUNT_)
		histogramVal = nbl_glsl_blit_normalization_getAlphaHistogramData(gl_LocalInvocationIndex, LAYER_IDX);

	const uint cumHistogramVal = nbl_glsl_workgroupInclusiveAdd(histogramVal);

	if (gl_LocalInvocationIndex < _NBL_GLSL_BLIT_ALPHA_BIN_COUNT_)
		scratchShared[gl_LocalInvocationIndex] = cumHistogramVal;
	barrier();

	const nbl_glsl_blit_parameters_t params = nbl_glsl_blit_getParameters();

	const uint outputPixelCount = params.outPixelCount;

	uint productMsb, productLsb;
	umulExtended(nbl_glsl_blit_normalization_getPassedInputPixelCountData(LAYER_IDX), outputPixelCount, productMsb, productLsb);

	const uint pixelsShouldPassCount = integerDivide_64_32_32(productMsb, productLsb, params.inPixelCount);
	const uint pixelsShouldFailCount = outputPixelCount - pixelsShouldPassCount;

	uint bucketIndex;
	{
		uint begin = 0u;
		const uint end = _NBL_GLSL_BLIT_ALPHA_BIN_COUNT_;
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

	const float newReferenceAlpha = min((bucketIndex - 0.5f) / float(_NBL_GLSL_BLIT_ALPHA_BIN_COUNT_ - 1), 1.f);

	const float alphaScale = params.referenceAlpha / newReferenceAlpha;

	const vec4 pixel = nbl_glsl_blit_normalization_getPaddedData(ivec3(gl_GlobalInvocationID));
	vec4 scaledPixel = vec4(pixel.rgb, pixel.a * alphaScale);
	nbl_glsl_blit_normalization_setData(scaledPixel, ivec3(gl_GlobalInvocationID));
}

#undef scratchShared

#define _NBL_GLSL_BLIT_NORMALIZATION_MAIN_DEFINED_
#endif

#endif
