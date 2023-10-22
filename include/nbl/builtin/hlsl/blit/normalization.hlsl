// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BLIT_NORMALIZATION_INCLUDED_
#define _NBL_BUILTIN_HLSL_BLIT_NORMALIZATION_INCLUDED_


#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/math/arithmetic.hlsl>
#include <nbl/builtin/hlsl/blit/parameters.hlsl>
// TODO: Remove when PR #519 is merged
#include <nbl/builtin/hlsl/blit/temp.hlsl>


namespace nbl
{
namespace hlsl
{
namespace blit
{

template <typename ConstevalParameters>
struct normalization_t
{
	uint32_t inPixelCount;
	uint32_t outPixelCount;
	float32_t referenceAlpha;
	uint16_t3 outDims;

	static normalization_t create(parameters_t params)
	{
		normalization_t normalization;

		normalization.inPixelCount = params.inPixelCount;
		normalization.outPixelCount = params.outPixelCount;
		normalization.referenceAlpha = params.referenceAlpha;
		normalization.outDims = params.outputDims;

		return normalization;
	}

	template<
		typename InCombinedSamplerAccessor,
		typename OutCombinedSamplerAccessor,
		typename HistogramAccessor,
        typename PassedAlphaTestPixelsAccessor,
		typename SharedAccessor>
	void execute(
		NBL_CONST_REF_ARG(InCombinedSamplerAccessor) inCombinedSamplerAccessor,
		NBL_REF_ARG(OutCombinedSamplerAccessor) outCombinedSamplerAccessor,
		NBL_REF_ARG(HistogramAccessor) histogramAccessor,
        NBL_CONST_REF_ARG(PassedAlphaTestPixelsAccessor) passedAlphaTestPixelsAccessor,
		NBL_REF_ARG(SharedAccessor) sharedAccessor,
		NBL_REF_ARG(uint16_t3) workGroupID,
		NBL_REF_ARG(uint16_t3) globalInvocationID,
		uint16_t localInvocationIndex)
	{
		spirv::umul_result_t product = spirv::umulExtended(passedAlphaTestPixelsAccessor.get(workGroupID.z), outPixelCount);

		const uint32_t pixelsShouldPassCount = math::fastDivide_int23_t(product.msb, product.lsb, inPixelCount);
		const uint32_t pixelsShouldFailCount = outPixelCount - pixelsShouldPassCount;

		const uint32_t lastInvocationIndex = ConstevalParameters::WorkGroupSize - 1;
		const uint32_t previousInvocationIndex = (localInvocationIndex - 1) & lastInvocationIndex;

		if (localInvocationIndex == lastInvocationIndex)
			sharedAccessor.main.set(lastInvocationIndex, 0);

		for (uint32_t virtualInvocation = localInvocationIndex; virtualInvocation < ConstevalParameters::AlphaBinCount; virtualInvocation += ConstevalParameters::WorkGroupSize)
		{
			const uint32_t histogramVal = histogramAccessor.get(workGroupID.z, virtualInvocation);

			GroupMemoryBarrierWithGroupSync();

			const uint32_t previousBlockSum = sharedAccessor.main.get(lastInvocationIndex);

			// TODO: Once PR #519 is merged replace with this and delete the line
			//const uint32_t cumHistogramVal = workgroup::inclusive_scan<uint32_t, binops::add<uint32_t>>(histogramVal, sharedAccessor) + previousBlockSum;
			const uint32_t cumHistogramVal =
				workgroup::inclusive_scan<ConstevalParameters::WorkGroupSize, uint32_t, binops::add<uint32_t> >(histogramVal, sharedAccessor, uint32_t(localInvocationIndex)) + previousBlockSum;

			sharedAccessor.main.set(localInvocationIndex, cumHistogramVal);

			GroupMemoryBarrierWithGroupSync();

			if (pixelsShouldFailCount <= cumHistogramVal)
			{
				const uint32_t previousBucketVal = bool(localInvocationIndex) ? sharedAccessor.main.get(previousInvocationIndex) : previousBlockSum;
				if (pixelsShouldFailCount > previousBucketVal)
					sharedAccessor.main.set(ConstevalParameters::WorkGroupSize, virtualInvocation);
			}
		}
		GroupMemoryBarrierWithGroupSync();

		const uint32_t bucketIndex = sharedAccessor.main.get(ConstevalParameters::WorkGroupSize);
		const float32_t newReferenceAlpha = min((bucketIndex - 0.5f) / float32_t(ConstevalParameters::AlphaBinCount - 1), 1.f);
		const float32_t alphaScale = referenceAlpha / newReferenceAlpha;

		if (all(globalInvocationID < outDims))
		{
			const float32_t4 pixel = inCombinedSamplerAccessor.get(globalInvocationID, workGroupID.z);
			const float32_t4 scaledPixel = float32_t4(pixel.rgb, pixel.a * alphaScale);
			outCombinedSamplerAccessor.set(globalInvocationID, workGroupID.z, scaledPixel);
		}
	}
};


}
}
}

#endif
