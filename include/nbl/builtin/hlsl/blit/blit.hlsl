// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BLIT_INCLUDED_
#define _NBL_BUILTIN_HLSL_BLIT_INCLUDED_


#include <nbl/builtin/hlsl/multi_dimensional_array_addressing/multi_dimensional_array_addressing.hlsl>
#include <nbl/builtin/hlsl/blit/parameters.hlsl>
#include <nbl/builtin/hlsl/blit/common.hlsl>
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/core.hlsl>

namespace nbl
{
namespace hlsl
{
namespace blit
{

// p in input space
int32_t3 getMinKernelWindowCoord(NBL_CONST_REF_ARG(int32_t3) p, NBL_CONST_REF_ARG(float3) minSupport)
{
	return int32_t3(ceil(p - float3(0.5f, 0.5f, 0.5f) + minSupport));
}

template <
	uint32_t SMemFloatsPerChannel,
	uint32_t BlitOutChannelCount,
	uint32_t BlitDimCount,
	uint32_t AlphaBinCount,
	uint32_t WorkGroupSize,
	E_FORMAT SoftwareEncodeFormat,
	typename InTexture,
	typename OutTexture,
	typename KernelWeights,
	typename Statistics,
	typename SharedAccessor>
void blit(
	NBL_CONST_REF_ARG(InTexture) inTex,
	NBL_REF_ARG(OutTexture) outTex,
	NBL_CONST_REF_ARG(KernelWeights) kernelWeights,
	NBL_REF_ARG(Statistics) statistics,
	NBL_CONST_REF_ARG(SharedAccessor) sharedAccessor,
	NBL_CONST_REF_ARG(parameters_t) params,
	uint32_t3 groupID,
	uint32_t groupIndex)
{
	const uint32_t3 inDim = params.getInputImageDimensions();
	const uint32_t3 outDim = params.getOutputImageDimensions();

	const uint32_t3 windowDim = params.getWindowDimensions();
	const uint32_t3 phaseCount = params.getPhaseCount();

	const uint32_t3 outputTexelsPerWG = params.getOutputTexelsPerWG();


	const float3 scale = params.fScale;
	const float3 halfScale = scale * float3(0.5f, 0.5f, 0.5f);

	const uint32_t3 minOutputPixel = groupID * outputTexelsPerWG;
	const float3 minOutputPixelCenterOfWG = float3(minOutputPixel)*scale + halfScale;
	const int32_t3 regionStartCoord = getMinKernelWindowCoord(minOutputPixelCenterOfWG, params.negativeSupport); // this can be negative, in which case HW sampler takes care of wrapping for us

	const uint32_t3 preloadRegion = params.preloadRegion;

	const uint32_t virtualInvocations = preloadRegion.x * preloadRegion.y * preloadRegion.z;
	for (uint32_t virtualInvocation = groupIndex; virtualInvocation < virtualInvocations; virtualInvocation += WorkGroupSize)
	{
		const int32_t3 inputPixelCoord = regionStartCoord + int32_t3(multi_dimensional_array_addressing::snakeCurveInverse(virtualInvocation, preloadRegion));

		const float4 loadedData = getData<BlitDimCount>(inTex, inputPixelCoord, groupID.z);
		for (uint32_t ch = 0; ch < BlitOutChannelCount; ++ch)
			sharedAccessor.set(ch * SMemFloatsPerChannel + virtualInvocation, loadedData[ch]);
	}
	glsl::barrier();

	const uint32_t3 iterationRegionPrefixProducts[3] = { params.iterationRegionXPrefixProducts, params.iterationRegionYPrefixProducts, params.iterationRegionZPrefixProducts};

	uint32_t readScratchOffset = 0;
	uint32_t writeScratchOffset = params.secondScratchOffset;
	for (uint32_t axis = 0; axis < BlitDimCount; ++axis)
	{
		for (uint32_t virtualInvocation = groupIndex; virtualInvocation < iterationRegionPrefixProducts[axis].z; virtualInvocation += WorkGroupSize)
		{
			const uint32_t3 virtualInvocationID = multi_dimensional_array_addressing::snakeCurveInverse(virtualInvocation, iterationRegionPrefixProducts[axis].xy);

			uint32_t outputPixel = virtualInvocationID.x;
			if (axis == 2)
				outputPixel = virtualInvocationID.z;
			outputPixel += minOutputPixel[axis];

			if (outputPixel >= outDim[axis])
				break;

			const int32_t minKernelWindow = int32_t(ceil((outputPixel + 0.5f) * scale[axis] - 0.5f + params.negativeSupport[axis]));

			// Combined stride for the two non-blitting dimensions, tightly coupled and experimentally derived with/by `iterationRegionPrefixProducts` above and the general order of iteration we use to avoid
			// read bank conflicts.
			uint32_t combinedStride;
			{
				if (axis == 0)
					combinedStride = virtualInvocationID.z * preloadRegion.y + virtualInvocationID.y;
				else if (axis == 1)
					combinedStride = virtualInvocationID.z * outputTexelsPerWG.x + virtualInvocationID.y;
				else if (axis == 2)
					combinedStride = virtualInvocationID.y * outputTexelsPerWG.y + virtualInvocationID.x;
			}

			uint32_t offset = readScratchOffset + (minKernelWindow - regionStartCoord[axis]) + combinedStride*preloadRegion[axis];
			const uint32_t windowPhase = outputPixel % phaseCount[axis];

			uint32_t kernelWeightIndex;
			if (axis == 0)
				kernelWeightIndex = windowPhase * windowDim.x;
			else if (axis == 1)
				kernelWeightIndex = params.kernelWeightsOffsetY + windowPhase * windowDim.y;
			else if (axis == 2)
				kernelWeightIndex = params.kernelWeightsOffsetZ + windowPhase * windowDim.z;

			float4 kernelWeight = kernelWeights[kernelWeightIndex];

			float4 accum = float4(0.f, 0.f, 0.f, 0.f);
			for (uint32_t ch = 0; ch < BlitOutChannelCount; ++ch)
				accum[ch] = sharedAccessor.get(ch * SMemFloatsPerChannel + offset) * kernelWeight[ch];

			for (uint32_t i = 1; i < windowDim[axis]; ++i)
			{
				kernelWeightIndex++;
				offset++;

				kernelWeight = kernelWeights[kernelWeightIndex];
				for (uint ch = 0; ch < BlitOutChannelCount; ++ch)
					accum[ch] += sharedAccessor.get(ch * SMemFloatsPerChannel + offset) * kernelWeight[ch];
			}

			const bool lastPass = (axis == (BlitDimCount - 1));
			if (lastPass)
			{
				// Tightly coupled with iteration order (`iterationRegionPrefixProducts`)
				uint32_t3 outCoord = virtualInvocationID.yxz;
				if (axis == 0)
					outCoord = virtualInvocationID.xyz;
				outCoord += minOutputPixel;

				const uint32_t bucketIndex = uint32_t(round(clamp(accum.a, 0, 1) * float(AlphaBinCount-1)));
				InterlockedAdd(statistics[bucketIndex].histogram[groupID.z], uint32_t(1));

				setData<BlitDimCount, SoftwareEncodeFormat>(outTex, outCoord, groupID.z, accum);
			}
			else
			{
				uint32_t scratchOffset = writeScratchOffset;
				if (axis == 0)
					scratchOffset += multi_dimensional_array_addressing::snakeCurve(virtualInvocationID.yxz, uint32_t3(preloadRegion.y, outputTexelsPerWG.x, preloadRegion.z));
				else
					scratchOffset += writeScratchOffset + multi_dimensional_array_addressing::snakeCurve(virtualInvocationID.zxy, uint32_t3(preloadRegion.z, outputTexelsPerWG.y, outputTexelsPerWG.x));

				for (uint32_t ch = 0; ch < BlitOutChannelCount; ++ch)
					sharedAccessor.set(ch * SMemFloatsPerChannel + scratchOffset, accum[ch]);
			}
		}

		const uint32_t tmp = readScratchOffset;
		readScratchOffset = writeScratchOffset;
		writeScratchOffset = tmp;
		glsl::barrier();
	}
}

}
}
}
#endif