// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BLIT_INCLUDED_
#define _NBL_BUILTIN_HLSL_BLIT_INCLUDED_


#include <nbl/builtin/hlsl/ndarray_addressing.hlsl>
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/blit/parameters.hlsl>


namespace nbl
{
namespace hlsl
{
namespace blit
{

template<
	bool DoCoverage,
	uint16_t WorkGroupSize,
	int32_t Dims,
	typename InCombinedSamplerAccessor,
	typename OutImageAccessor,
//	typename KernelWeightsAccessor,
//	typename HistogramAccessor,
	typename SharedAccessor
>
void execute(
	NBL_CONST_REF_ARG(InCombinedSamplerAccessor) inCombinedSamplerAccessor,
	NBL_REF_ARG(OutImageAccessor) outImageAccessor,
//	NBL_CONST_REF_ARG(KernelWeightsAccessor) kernelWeightsAccessor,
//	NBL_REF_ARG(HistogramAccessor) histogramAccessor,
	NBL_REF_ARG(SharedAccessor) sharedAccessor,
	NBL_CONST_REF_ARG(SPerWorkgroup) params,
	const uint16_t layer,
	const vector<uint16_t,Dims> virtWorkGroupID
)
{
	using uint16_tN = vector<uint16_t,Dims>;
	// the dimensional truncation is desired
	const uint16_tN outputTexelsPerWG = uint16_tN(params.getOutputBaseCoord(uint16_t3(1,1,1)));
	// its the min XYZ corner of the area the workgroup will sample from to produce its output
	const uint16_tN minOutputPixel = virtWorkGroupID*outputTexelsPerWG;

	using float32_tN = vector<float32_t,Dims>;
	const float32_tN scale = _static_cast<float32_tN>(params.scale);
	const float32_tN lastInputTexel = _static_cast<float32_tN>(params.getInputLastTexel());
	const uint16_t inLevel = _static_cast<uint16_t>(params.inLevel);
	const float32_tN inImageSizeRcp = inCombinedSamplerAccessor.template extentRcp<Dims>(inLevel);

	using int32_tN = vector<int32_t,Dims>;
	// intermediate result only needed to compute `regionStartCoord`, basically the sampling coordinate of the minOutputPixel in the input texture
	const float32_tN noGoodNameForThisThing = (float32_tN(minOutputPixel)+promote<float32_tN>(0.5f))*scale-promote<float32_tN>(0.5f);
	// can be negative, its the min XYZ corner of the area the workgroup will sample from to produce its output
	// TODO: is there a HLSL/SPIR-V round() that can simplify ceil(x-0.5)+0.5 ?
	const float32_tN regionStartCoord = ceil(noGoodNameForThisThing)+promote<float32_tN>(0.5f);
	const float32_tN regionNextStartCoord = ceil(noGoodNameForThisThing+float32_tN(outputTexelsPerWG)*scale)+promote<float32_tN>(0.5f);

	const uint16_tN preloadRegion = _static_cast<uint16_tN>(params.getPreloadExtentExceptLast());
	const uint16_t localInvocationIndex = _static_cast<uint16_t>(glsl::gl_LocalInvocationIndex()); // workgroup::SubgroupContiguousIndex()
	// need to clear our atomic coverage counter to 0 
	const uint16_t coverageDWORD = _static_cast<uint16_t>(params.coverageDWORD);
	if (DoCoverage)
	{
		if (localInvocationIndex==0)
			sharedAccessor.set(coverageDWORD,0u);
		glsl::barrier();
	}
	const uint16_t preloadLast = _static_cast<uint16_t>(params.preloadLast);
	for (uint16_t virtualInvocation=localInvocationIndex; virtualInvocation<=preloadLast; virtualInvocation+=WorkGroupSize)
	{
		// if we make all args in snakeCurveInverse 16bit maybe compiler will optimize the divisions into using float32_t
		const uint16_tN virtualInvocationID = ndarray_addressing::snakeCurveInverse<Dims,uint16_t,uint16_t>(virtualInvocation,preloadRegion);
		const float32_tN inputTexCoordUnnorm = regionStartCoord + float32_tN(virtualInvocationID);
		const float32_tN inputTexCoord = inputTexCoordUnnorm * inImageSizeRcp;

		const float32_t4 loadedData = inCombinedSamplerAccessor.template get<float32_t,Dims>(inputTexCoord,layer,inLevel);

		if (DoCoverage)
		if (loadedData[params.coverageChannel]>=params.alphaRefValue &&
			all(inputTexCoordUnnorm<regionNextStartCoord) && // not overlapping with the next tile
			all(inputTexCoord>=promote<float32_tN>(0.5f)) && // within the image from below
			all(inputTexCoordUnnorm<=lastInputTexel) // within the image from above
		)
		{
			sharedAccessor.template atomicIncr<uint32_t>(coverageDWORD);
		}

		[unroll(4)]
		for (uint16_t ch=0; ch<4 && ch<=params.lastChannel; ch++)
			sharedAccessor.set(preloadLast*ch+ch+virtualInvocation,loadedData[ch]);
	}
	glsl::barrier();

	uint16_t readScratchOffset = uint16_t(0);
	uint16_t writeScratchOffset = _static_cast<uint16_t>(params.secondScratchOffset);
	uint16_tN currentOutRegion = preloadRegion;
	currentOutRegion[0] = outputTexelsPerWG[0];
	[unroll(3)]
	for (int32_t axis=0; axis<Dims; axis++)
	for (uint16_t virtualInvocation=localInvocationIndex; virtualInvocation<=0x45; virtualInvocation+=WorkGroupSize)
	{
		// this always maps to the index in the current pass output
		const uint16_tN virtualInvocationID = ndarray_addressing::snakeCurveInverse<Dims,uint16_t,uint16_t>(virtualInvocation,currentOutRegion); 

		//
	}
/*
	for (uint16_t virtualInvocation=localInvocationIndex; virtualInvocation<outputTexelsPerWG.x*outputTexelsPerWG.y*outputTexelsPerWG.z; virtualInvocation+=WorkGroupSize)
	{
		float32_t4 fullValue;
		[unroll(4)]
		for (uint16_t ch=0; ch<4 && ch<=params.lastChannel; ch++)
		{
			float32_t value; // TODO
			if (DoCoverage && ch==params.coverageChannel)
			{
				// TODO: global histogram increment
			}
			fullValue[ch] = value;
		}
		outImageAccessor.set(minOutputPixel,layer,fullValue);
	}
*/
}

#if 0
template <typename ConstevalParameters>
struct compute_blit_t
{
	float32_t3 scale;
	float32_t3 negativeSupport;
	uint32_t kernelWeightsOffsetY;
	uint32_t kernelWeightsOffsetZ;
	uint32_t inPixelCount;
	uint32_t outPixelCount;
	uint16_t3 inDims;
	uint16_t3 outDims;
	uint16_t3 windowDims;
	uint16_t3 phaseCount;
	uint16_t3 preloadRegion;
	uint16_t3 iterationRegionXPrefixProducts;
	uint16_t3 iterationRegionYPrefixProducts;
	uint16_t3 iterationRegionZPrefixProducts;
	uint16_t secondScratchOffset;

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
		// bottom of the input tile
		const uint32_t3 minOutputPixel = workGroupID * outputTexelsPerWG;


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
#endif

}
}
}

#endif