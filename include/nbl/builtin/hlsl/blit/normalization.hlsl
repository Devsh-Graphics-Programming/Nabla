// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BLIT_NORMALIZATION_INCLUDED_
#define _NBL_BUILTIN_HLSL_BLIT_NORMALIZATION_INCLUDED_

#include <nbl/builtin/hlsl/workgroup/arithmetic.hlsl>
#include <nbl/builtin/hlsl/blit/parameters.hlsl>
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>

namespace nbl
{
namespace hlsl
{
namespace blit
{


//! WARNING: ONLY WORKS FOR `dividendMsb<=2^23` DUE TO FP32 ABUSE !!!
uint32_t integerDivide_64_32_32(in uint32_t dividendMsb, in uint32_t dividendLsb, in uint32_t divisor)
{
	const float msbRatio = float(dividendMsb) / float(divisor);
	const uint32_t quotient = uint32_t((msbRatio * (~0u)) + msbRatio) + dividendLsb / divisor;
	return quotient;
}

struct OpAdd
{
	uint operator()(uint v1, uint v2) { return v1 + v2; }
};

template <
	uint32_t WorkGroupSize,
	uint32_t BlitDimCount,
	uint32_t AlphaBinCount,
	E_FORMAT SoftwareEncodeFormat,
	typename InTexture,
	typename OutTexture,
	typename Statistics,
	typename SharedAccessor>
void normalization(
	NBL_CONST_REF_ARG(InTexture) inTexure,
	NBL_REF_ARG(OutTexture) outTexure,
	NBL_REF_ARG(Statistics) statistics,
	NBL_CONST_REF_ARG(SharedAccessor) sharedAccessor,
	NBL_CONST_REF_ARG(parameters_t) params,
	NBL_REF_ARG(uint32_t3) groupID,
	NBL_REF_ARG(uint32_t3) dispatchThreadID,
	uint32_t groupIndex)
{
	const uint32_t outputPixelCount = params.outPixelCount;

	// TODO: Replace with intrinsic when those headers are finalized.
	// This assumes that inside spirv_intrinsics/arithmetic.hlsl there is something like:
	//[[vk::ext_instruction(/* OpUMulExtended */ 151)]]
	//pair<uint32_t, uint32_t> umulExtended(uint32_t v0, uint32_t v1);
	pair<uint32_t, uint32_t> product = spirv::umulExtended(statistics[groupID.z].passedPixelCount, outputPixelCount);

	const uint32_t pixelsShouldPassCount = integerDivide_64_32_32(product.second, product.first, params.inPixelCount);
	const uint32_t pixelsShouldFailCount = outputPixelCount - pixelsShouldPassCount;

	const uint32_t lastInvocationIndex = WorkGroupSize - 1;
	const uint32_t previousInvocationIndex = (groupIndex - 1) & lastInvocationIndex;

	if (groupIndex == lastInvocationIndex)
		sharedAccessor.set(lastInvocationIndex, 0);

	for (uint32_t virtualInvocation = groupIndex; virtualInvocation < AlphaBinCount; virtualInvocation += _NBL_GLSL_WORKGROUP_SIZE_)
	{
		const uint32_t histogramVal = statistics[virtualInvocation].histogram[groupID.z];

		glsl::barrier();

		const uint32_t previousBlockSum = sharedAccessor.get(lastInvocationIndex);

		const uint32_t cumHistogramVal = workgroup::inclusive_scan<uint32_t, OpAdd>(histogramVal, sharedAccessor) + previousBlockSum;

		sharedAccessor.set(groupIndex, cumHistogramVal);

		glsl::barrier();

		if (pixelsShouldFailCount <= cumHistogramVal)
		{
			const uint32_t previousBucketVal = bool(groupIndex) ? sharedAccessor.get(previousInvocationIndex) : previousBlockSum;
			if (pixelsShouldFailCount > previousBucketVal)
				sharedAccessor.set(sharedAccessor.size() - 1, virtualInvocation);
		}
	}
	glsl::barrier();

	const uint32_t bucketIndex = sharedAccessor.get(sharedAccessor.size() - 1);
	const float newReferenceAlpha = min((bucketIndex - 0.5f) / float(AlphaBinCount - 1), 1.f);
	const float alphaScale = params.referenceAlpha / newReferenceAlpha;

	const uint32_t3 outDim = params.getOutputImageDimensions();

	if (all(dispatchThreadID < outDim))
	{
		const float4 pixel = getData<BlitDimCount>(inTexture, dispatchThreadID, groupID.z);
		const float4 scaledPixel = float4(pixel.rgb, pixel.a * alphaScale);
		setData<BlitDimCount, SoftwareEncodeFormat>(outTexture, scaledPixel, dispatchThreadID, groupID.z);
	}
}

}
}
}
#endif
