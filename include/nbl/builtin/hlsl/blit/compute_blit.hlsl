// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BLIT_INCLUDED_
#define _NBL_BUILTIN_HLSL_BLIT_INCLUDED_


#include <nbl/builtin/hlsl/ndarray_addressing.hlsl>
#include <nbl/builtin/hlsl/blit/parameters.hlsl>
#include <nbl/builtin/hlsl/blit/common.hlsl>


namespace nbl
{
namespace hlsl
{
namespace blit
{

template <typename ConstevalParameters>
struct compute_blit_t
{
	float32_t3 scale;
	float32_t3 negativeSupport;
	uint32_t kernelWeightsOffsetY;
	uint32_t kernelWeightsOffsetZ;
	uint32_t inPixelCount;
	uint32_t outPixelCount;
	uint16_t3 outputTexelsPerWG;
	uint16_t3 inDims;
	uint16_t3 outDims;
	uint16_t3 windowDims;
	uint16_t3 phaseCount;
	uint16_t3 preloadRegion;
	uint16_t3 iterationRegionXPrefixProducts;
	uint16_t3 iterationRegionYPrefixProducts;
	uint16_t3 iterationRegionZPrefixProducts;
	uint16_t secondScratchOffset;

	static compute_blit_t create(NBL_CONST_REF_ARG(parameters_t) params)
	{
		compute_blit_t compute_blit;

		compute_blit.scale = params.fScale;
		compute_blit.negativeSupport = params.negativeSupport;
		compute_blit.kernelWeightsOffsetY = params.kernelWeightsOffsetY;
		compute_blit.kernelWeightsOffsetZ = params.kernelWeightsOffsetZ;
		compute_blit.inPixelCount = params.inPixelCount;
		compute_blit.outPixelCount = params.outPixelCount;
		compute_blit.outputTexelsPerWG = params.getOutputTexelsPerWG();
		compute_blit.inDims = params.inputDims;
		compute_blit.outDims = params.outputDims;
		compute_blit.windowDims = params.windowDims;
		compute_blit.phaseCount = params.phaseCount;
		compute_blit.preloadRegion = params.preloadRegion;
		compute_blit.iterationRegionXPrefixProducts = params.iterationRegionXPrefixProducts;
		compute_blit.iterationRegionYPrefixProducts = params.iterationRegionYPrefixProducts;
		compute_blit.iterationRegionZPrefixProducts = params.iterationRegionZPrefixProducts;
		compute_blit.secondScratchOffset = params.secondScratchOffset;

		return compute_blit;
	}

	template <
		typename InCombinedSamplerAccessor,
		typename OutImageAccessor,
		typename KernelWeightsAccessor,
		typename HistogramAccessor,
		typename SharedAccessor>
	void execute(
		NBL_CONST_REF_ARG(InCombinedSamplerAccessor) inCombinedSamplerAccessor,
		NBL_REF_ARG(OutImageAccessor) outImageAccessor,
		NBL_CONST_REF_ARG(KernelWeightsAccessor) kernelWeightsAccessor,
		NBL_REF_ARG(HistogramAccessor) histogramAccessor,
		NBL_REF_ARG(SharedAccessor) sharedAccessor,
		uint16_t3 workGroupID,
		uint16_t localInvocationIndex)
	{
		const float3 halfScale = scale * float3(0.5f, 0.5f, 0.5f);
		// bottom of the input tile
		const uint32_t3 minOutputPixel = workGroupID * outputTexelsPerWG;
		const float3 minOutputPixelCenterOfWG = float3(minOutputPixel)*scale + halfScale;
		// this can be negative, in which case HW sampler takes care of wrapping for us
		const int32_t3 regionStartCoord = int32_t3(ceil(minOutputPixelCenterOfWG - float3(0.5f, 0.5f, 0.5f) + negativeSupport));

		const uint32_t virtualInvocations = preloadRegion.x * preloadRegion.y * preloadRegion.z;
		for (uint32_t virtualInvocation = localInvocationIndex; virtualInvocation < virtualInvocations; virtualInvocation += ConstevalParameters::WorkGroupSize)
		{
			const int32_t3 inputPixelCoord = regionStartCoord + int32_t3(ndarray_addressing::snakeCurveInverse(virtualInvocation, preloadRegion));
			float32_t3 inputTexCoord = (inputPixelCoord + float32_t3(0.5f, 0.5f, 0.5f)) / inDims;
			const float4 loadedData = inCombinedSamplerAccessor.get(inputTexCoord, workGroupID.z);

			for (uint32_t ch = 0; ch < ConstevalParameters::BlitOutChannelCount; ++ch)
				sharedAccessor.set(ch * ConstevalParameters::SMemFloatsPerChannel + virtualInvocation, loadedData[ch]);
		}
		GroupMemoryBarrierWithGroupSync();

		const uint32_t3 iterationRegionPrefixProducts[3] = {iterationRegionXPrefixProducts, iterationRegionYPrefixProducts, iterationRegionZPrefixProducts};

		uint32_t readScratchOffset = 0;
		uint32_t writeScratchOffset = secondScratchOffset;
		for (uint32_t axis = 0; axis < ConstevalParameters::BlitDimCount; ++axis)
		{
			for (uint32_t virtualInvocation = localInvocationIndex; virtualInvocation < iterationRegionPrefixProducts[axis].z; virtualInvocation += ConstevalParameters::WorkGroupSize)
			{
				const uint32_t3 virtualInvocationID = ndarray_addressing::snakeCurveInverse(virtualInvocation, iterationRegionPrefixProducts[axis].xy);

				uint32_t outputPixel = virtualInvocationID.x;
				if (axis == 2)
					outputPixel = virtualInvocationID.z;
				outputPixel += minOutputPixel[axis];

				if (outputPixel >= outDims[axis])
					break;

				const int32_t minKernelWindow = int32_t(ceil((outputPixel + 0.5f) * scale[axis] - 0.5f + negativeSupport[axis]));

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
					kernelWeightIndex = windowPhase * windowDims.x;
				else if (axis == 1)
					kernelWeightIndex = kernelWeightsOffsetY + windowPhase * windowDims.y;
				else if (axis == 2)
					kernelWeightIndex = kernelWeightsOffsetZ + windowPhase * windowDims.z;

				float4 kernelWeight = kernelWeightsAccessor.get(kernelWeightIndex);

				float4 accum = float4(0.f, 0.f, 0.f, 0.f);
				for (uint32_t ch = 0; ch < ConstevalParameters::BlitOutChannelCount; ++ch)
					accum[ch] = sharedAccessor.get(ch * ConstevalParameters::SMemFloatsPerChannel + offset) * kernelWeight[ch];

				for (uint32_t i = 1; i < windowDims[axis]; ++i)
				{
					kernelWeightIndex++;
					offset++;

					kernelWeight = kernelWeightsAccessor.get(kernelWeightIndex);
					for (uint ch = 0; ch < ConstevalParameters::BlitOutChannelCount; ++ch)
						accum[ch] += sharedAccessor.get(ch * ConstevalParameters::SMemFloatsPerChannel + offset) * kernelWeight[ch];
				}

				const bool lastPass = (axis == (ConstevalParameters::BlitDimCount - 1));
				if (lastPass)
				{
					// Tightly coupled with iteration order (`iterationRegionPrefixProducts`)
					uint32_t3 outCoord = virtualInvocationID.yxz;
					if (axis == 0)
						outCoord = virtualInvocationID.xyz;
					outCoord += minOutputPixel;

					const uint32_t bucketIndex = uint32_t(round(clamp(accum.a, 0, 1) * float(ConstevalParameters::AlphaBinCount-1)));
					histogramAccessor.atomicAdd(workGroupID.z, bucketIndex, uint32_t(1));

					outImageAccessor.set(outCoord, workGroupID.z, accum);
				}
				else
				{
					uint32_t scratchOffset = writeScratchOffset;
					if (axis == 0)
						scratchOffset += ndarray_addressing::snakeCurve(virtualInvocationID.yxz, uint32_t3(preloadRegion.y, outputTexelsPerWG.x, preloadRegion.z));
					else
						scratchOffset += writeScratchOffset + ndarray_addressing::snakeCurve(virtualInvocationID.zxy, uint32_t3(preloadRegion.z, outputTexelsPerWG.y, outputTexelsPerWG.x));

					for (uint32_t ch = 0; ch < ConstevalParameters::BlitOutChannelCount; ++ch)
						sharedAccessor.set(ch * ConstevalParameters::SMemFloatsPerChannel + scratchOffset, accum[ch]);
				}
			}

			const uint32_t tmp = readScratchOffset;
			readScratchOffset = writeScratchOffset;
			writeScratchOffset = tmp;
			GroupMemoryBarrierWithGroupSync();
		}
	}
};

}
}
}

#endif