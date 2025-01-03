// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BLIT_INCLUDED_
#define _NBL_BUILTIN_HLSL_BLIT_INCLUDED_


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
	const uint16_t lastChannel = params.lastChannel;
	const uint16_t coverageChannel = params.coverageChannel;

	using uint16_tN = vector<uint16_t,Dims>;
	// the dimensional truncation is desired
	const uint16_tN outputTexelsPerWG = params.template getPerWGOutputExtent<Dims>();
	// its the min XYZ corner of the area the workgroup will sample from to produce its output
	const uint16_tN minOutputTexel = virtWorkGroupID*outputTexelsPerWG;

	using float32_tN = vector<float32_t,Dims>;
	const float32_tN scale = truncate<Dims>(params.scale);
	const float32_tN inputMaxCoord = params.template getInputMaxCoord<Dims>();
	const uint16_t inLevel = _static_cast<uint16_t>(params.inLevel);
	const float32_tN inImageSizeRcp = inCombinedSamplerAccessor.template extentRcp<Dims>(inLevel);

	using int32_tN = vector<int32_t,Dims>;
	// can be negative, its the min XYZ corner of the area the workgroup will sample from to produce its output
	const float32_tN regionStartCoord = params.inputUpperBound<Dims>(minOutputTexel);
	const float32_tN regionNextStartCoord = params.inputUpperBound<Dims>(minOutputTexel+outputTexelsPerWG);

	const uint16_t localInvocationIndex = _static_cast<uint16_t>(glsl::gl_LocalInvocationIndex()); // workgroup::SubgroupContiguousIndex()

	// need to clear our atomic coverage counter to 0 
	const uint16_t coverageDWORD = _static_cast<uint16_t>(params.coverageDWORD);
	if (DoCoverage)
	{
		if (localInvocationIndex==0)
			sharedAccessor.set(coverageDWORD,0u);
		glsl::barrier();
	}

	//
	const PatchLayout<Dims> preloadLayout = params.getPreloadMeta();
	for (uint16_t virtualInvocation=localInvocationIndex; virtualInvocation<preloadLayout.getLinearEnd(); virtualInvocation+=WorkGroupSize)
	{
		// if we make all args in snakeCurveInverse 16bit maybe compiler will optimize the divisions into using float32_t
		const uint16_tN virtualInvocationID = preloadLayout.getID(virtualInvocation);
		const float32_tN inputTexCoordUnnorm = regionStartCoord + float32_tN(virtualInvocationID);

		const float32_tN inputTexCoord = (inputTexCoordUnnorm + promote<float32_tN>(0.5f)) * inImageSizeRcp;
		const float32_t4 loadedData = inCombinedSamplerAccessor.template get<float32_t,Dims>(inputTexCoord,layer,inLevel);

		if (DoCoverage)
		if (loadedData[coverageChannel]>=params.alphaRefValue &&
			all(inputTexCoordUnnorm<regionNextStartCoord) && // not overlapping with the next tile
			all(inputTexCoordUnnorm>=promote<float32_tN>(0.f)) && // within the image from below
			all(inputTexCoordUnnorm<=inputMaxCoord) // within the image from above
		)
		{
			// TODO: atomicIncr or a workgroup reduction of ballots?
//			sharedAccessor.template atomicIncr<uint32_t>(coverageDWORD);
		}

		[unroll(4)]
		for (uint16_t ch=0; ch<4 && ch<=lastChannel; ch++)
			sharedAccessor.template set<float32_t>(preloadCount*ch+virtualInvocation,loadedData[ch]);
	}
	glsl::barrier();

	uint16_t readScratchOffset = uint16_t(0);
	uint16_t writeScratchOffset = _static_cast<uint16_t>(params.secondScratchOffDWORD);
	const uint16_tN windowExtent = params.template getWindowExtent<Dims>();
	uint16_t prevLayout = preloadLayout;
	uint32_t kernelWeightOffset = 0;
	[unroll(3)]
	for (int32_t axis=0; axis<Dims; axis++)
	{
		const PatchLayout<Dims> outputLayout = params.getPassMeta<Dims>(axis);
		const uint16_t invocationCount = outputLayout.getLinearEnd();
		const uint16_t phaseCount = params.getPhaseCount(axis);
		const uint16_t windowLength = windowExtent[axis];
		const uint16_t prevPassInvocationCount = prevLayout.getLinearEnd();
		for (uint16_t virtualInvocation=localInvocationIndex; virtualInvocation<invocationCount; virtualInvocation+=WorkGroupSize)
		{
			// this always maps to the index in the current pass output
			const uint16_tN virtualInvocationID = outputLayout.getID(virtualInvocation);

			// we sweep along a line at a time, `[0]` is not a typo, look at the definition of `params.getPassMeta`
			uint16_t localOutputCoord = virtualInvocationID[0];
			// we can actually compute the output position of this line
			const uint16_t globalOutputCoord = localOutputCoord+minOutputTexel[axis];
			// hopefull the compiler will see that float32_t may be possible here due to `sizeof(float32_t mantissa)>sizeof(uint16_t)`
			const uint32_t windowPhase = globalOutputCoord % phaseCount;

			//const int32_t windowStart = ceil(localOutputCoord+0.5f;

			// let us sweep
			float32_t4 accum = promote<float32_t4>(0.f);
			{
				uint32_t kernelWeightIndex = windowPhase*windowLength+kernelWeightOffset;
				// Need to use global coordinate because of ceil(x*scale) involvement
				uint16_tN tmp; tmp[0] = params.inputUpperBound(globalOutputCoord,axis)-regionStartCoord;
				[unroll(2)]
				for (int32_t i=1; i<Dims; i++)
					tmp[i] = virtualInvocationID[i];
				// initialize to the first gather texel in range of the window for the output
				uint16_t inputIndex = readScratchOffset+prevLayout.getIndex(tmp);
				for (uint16_t i=0; i<windowLength; i++,inputIndex++)
				{
					const float32_t4 kernelWeight = kernelWeightsAccessor.get(kernelWeightIndex++);
					[unroll(4)]
					for (uint16_t ch=0; ch<4 && ch<=lastChannel; ch++)
						accum[ch] += sharedAccessor.template get<float32_t>(ch*prevPassInvocationCount+inputIndex)*kernelWeight[ch];
				}
			}

			// now write outputs
			if (axis!=Dims-1) // not last pass
			{
				const uint32_t scratchOffset = writeScratchOffset+params.template getStorageIndex<Dims>(axis,virtualInvocationID);
				[unroll(4)]
				for (uint16_t ch=0; ch<4 && ch<=lastChannel; ch++)
					sharedAccessor.template set(ch*invocationCount+scratchOffset,accum[ch]);
			}
			else
			{
				const uint16_tN coord = SPerWorkgroup::unswizzle<Dims>(virtualInvocationID)+minOutputTexel;
				outImageAccessor.template set<float32_t,Dims>(coord,layer,accum);
				if (DoCoverage)
				{
//					const uint32_t bucketIndex = uint32_t(round(accum[coverageChannel] * float(ConstevalParameters::AlphaBinCount - 1)));
//					histogramAccessor.atomicAdd(workGroupID.z,bucketIndex,uint32_t(1));
//					intermediateAlphaImageAccessor.template set<float32_t,Dims>(coord,layer,accum);
				}
			}
		}
		glsl::barrier();
		kernelWeightOffset += phaseCount*windowExtent;
		prevLayout = outputLayout;
		// TODO: use Przemog's `nbl::hlsl::swap` method when the float64 stuff gets merged
		const uint32_t tmp = readScratchOffset;
		readScratchOffset = writeScratchOffset;
		writeScratchOffset = tmp;
	}
}

}
}
}

#endif