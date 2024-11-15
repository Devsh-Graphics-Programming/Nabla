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
	const uint16_t lastChannel = params.lastChannel;
	const uint16_t coverageChannel = params.coverageChannel;

	using uint16_tN = vector<uint16_t,Dims>;
	// the dimensional truncation is desired
	const uint16_tN outputTexelsPerWG = truncate<Dims>(params.getOutputBaseCoord(uint16_t3(1,1,1)));
	// its the min XYZ corner of the area the workgroup will sample from to produce its output
	const uint16_tN minOutputTexel = virtWorkGroupID*outputTexelsPerWG;

	using float32_tN = vector<float32_t,Dims>;
	const float32_tN scale = truncate<Dims>(params.scale);
	const float32_tN inputEndCoord = truncate<Dims>(params.getInputEndCoord());
	const uint16_t inLevel = _static_cast<uint16_t>(params.inLevel);
	const float32_tN inImageSizeRcp = inCombinedSamplerAccessor.template extentRcp<Dims>(inLevel);

	using int32_tN = vector<int32_t,Dims>;
	// intermediate result only needed to compute `regionStartCoord`, basically the sampling coordinate of the minOutputTexel in the input texture
	const float32_tN noGoodNameForThisThing = (float32_tN(minOutputTexel)+promote<float32_tN>(0.5f))*scale-promote<float32_tN>(0.5f);
	// can be negative, its the min XYZ corner of the area the workgroup will sample from to produce its output
	// TODO: is there a HLSL/SPIR-V round() that can simplify ceil(x-0.5)+0.5 ?
	const float32_tN regionStartCoord = ceil(noGoodNameForThisThing)+promote<float32_tN>(0.5f);
	const float32_tN regionNextStartCoord = ceil(noGoodNameForThisThing+float32_tN(outputTexelsPerWG)*scale)+promote<float32_tN>(0.5f);

	const uint16_tN preloadRegion = truncate<Dims>(params.getPreloadExtentExceptLast());
	const uint16_t localInvocationIndex = _static_cast<uint16_t>(glsl::gl_LocalInvocationIndex()); // workgroup::SubgroupContiguousIndex()
	// need to clear our atomic coverage counter to 0 
	const uint16_t coverageDWORD = _static_cast<uint16_t>(params.coverageDWORD);
	if (DoCoverage)
	{
		if (localInvocationIndex==0)
			sharedAccessor.set(coverageDWORD,0u);
		glsl::barrier();
	}
	const uint16_t preloadCount = _static_cast<uint16_t>(params.preloadCount);
	for (uint16_t virtualInvocation=localInvocationIndex; virtualInvocation<preloadCount; virtualInvocation+=WorkGroupSize)
	{
		// if we make all args in snakeCurveInverse 16bit maybe compiler will optimize the divisions into using float32_t
		const uint16_tN virtualInvocationID = ndarray_addressing::snakeCurveInverse<Dims,uint16_t,uint16_t>(virtualInvocation,preloadRegion);
		const float32_tN inputTexCoordUnnorm = regionStartCoord + float32_tN(virtualInvocationID);
		const float32_tN inputTexCoord = inputTexCoordUnnorm * inImageSizeRcp;

		const float32_t4 loadedData = inCombinedSamplerAccessor.template get<float32_t,Dims>(inputTexCoord,layer,inLevel);

		if (DoCoverage)
		if (loadedData[coverageChannel]>=params.alphaRefValue &&
			all(inputTexCoordUnnorm<regionNextStartCoord) && // not overlapping with the next tile
			all(inputTexCoord>=promote<float32_tN>(0.5f)) && // within the image from below
			all(inputTexCoordUnnorm<inputMaxCoord) // within the image from above
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
	uint16_t writeScratchOffset = _static_cast<uint16_t>(params.secondScratchOffset);
	uint16_t prevPassInvocationCount = preloadCount;
	uint32_t kernelWeightOffset = 0;
	//
	uint16_tN currentOutRegion = preloadRegion;
	[unroll(3)]
	for (int32_t axis=0; axis<Dims; axis++)
	{
		const uint16_t phaseCount = params.getPhaseCount(axis);
		const uint32_t windowExtent = 0x45;
		// We sweep along X, then Y, then Z, at every step we need the loads from smem to be performed on consecutive values so that we don't have bank conflicts
		currentOutRegion[axis] = outputTexelsPerWG[axis];
		//
		const uint16_t invocationCount = params.getPassInvocationCount(axis);
		for (uint16_t virtualInvocation=localInvocationIndex; virtualInvocation<invocationCount; virtualInvocation+=WorkGroupSize)
		{
			// this always maps to the index in the current pass output
			const uint16_tN virtualInvocationID = ndarray_addressing::snakeCurveInverse<Dims,uint16_t,uint16_t>(virtualInvocation,currentOutRegion); 

			// we sweep along a line at a time
			uint16_t localOutputCoord = virtualInvocation[0]; // TODO
			// we can actually compute the output position of this line
			const uint16_t globalOutputCoord = localOutputCoord+minOutputTexel[axis];
			// hopefull the compiler will see that float32_t may be possible here due to sizeof(float32_t mantissa)>sizeof(uint16_t)
			const uint32_t windowPhase = globalOutputCoord % phaseCount;

			//const int32_t windowStart = ceil(localOutputCoord+0.5f;

			// let us sweep
			float32_t4 accum = promote<float32_t4>(0.f);
			{
				uint32_t kernelWeightIndex = windowPhase*windowExtent+kernelWeightOffset;
				uint16_t inputIndex = readScratchOffset+0x45; // (minKernelWindow - regionStartCoord[axis]) + combinedStride*preloadRegion[axis];
				for (uint16_t i=0; i<windowExtent; i++,inputIndex++)
				{
					const float32_t4 kernelWeight = kernelWeightsAccessor.get(kernelWeightIndex++);
					[unroll(4)]
					for (uint16_t ch=0; ch<4 && ch<=lastChannel; ch++)
						accum[ch] += sharedAccessor.template get<float32_t>(ch*prevPassInvocationCount+inputIndex)*kernelWeight[ch];
				}
			}

			// now write outputs
			if (axis<Dims-1) // not last pass
			{
				// Tightly coupled with iteration order (`iterationRegionPrefixProducts`)
				uint16_tN outCoord = virtualInvocationID.yxz;
				if (axis == 0)
					outCoord = virtualInvocationID.xyz;
				outCoord += minOutputTexel;


				outImageAccessor.template set<float32_t,Dims>(outCoord,layer,accum);
				if (DoCoverage)
				{
//					const uint32_t bucketIndex = uint32_t(round(accum[coverageChannel] * float(ConstevalParameters::AlphaBinCount - 1)));
//					histogramAccessor.atomicAdd(workGroupID.z,bucketIndex,uint32_t(1));
				}
			}
			else
			{
				uint32_t scratchOffset = writeScratchOffset;
				if (axis == 0)
					scratchOffset += ndarray_addressing::snakeCurve(virtualInvocationID.yxz, uint32_t3(preloadRegion.y, outputTexelsPerWG.x, preloadRegion.z));
				else
					scratchOffset += writeScratchOffset + ndarray_addressing::snakeCurve(virtualInvocationID.zxy, uint32_t3(preloadRegion.z, outputTexelsPerWG.y, outputTexelsPerWG.x));
				
				[unroll(4)]
				for (uint16_t ch=0; ch<4 && ch<=lastChannel; ch++)
					sharedAccessor.template set(ch*invocationCount+scratchOffset,accum[ch]);
			}
		}
		glsl::barrier();
		kernelWeightOffset += phaseCount*windowExtent;
		prevPassInvocationCount = invocationCount;
		// TODO: use Przemog's `nbl::hlsl::swap` method when the float64 stuff gets merged
		const uint32_t tmp = readScratchOffset;
		readScratchOffset = writeScratchOffset;
		writeScratchOffset = tmp;
	}
/*
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
		outImageAccessor.set(minOutputTexel,layer,fullValue);
*/
}

#if 0
				uint32_t outputPixel = virtualInvocationID.x;
				if (axis == 2)
					outputPixel = virtualInvocationID.z;
				outputPixel += minOutputPixel[axis];

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

			}
#endif

}
}
}

#endif