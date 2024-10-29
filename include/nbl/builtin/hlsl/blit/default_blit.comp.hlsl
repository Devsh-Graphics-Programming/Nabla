// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
//#include "nbl/builtin/hlsl/blit/common.hlsl"
//#include "nbl/builtin/hlsl/blit/parameters.hlsl"
//#include "nbl/builtin/hlsl/blit/compute_blit.hlsl"


groupshared uint32_t sMem[ConstevalParameters::SharedMemoryDWORDs];
/*
struct HistogramAccessor
{
	void atomicAdd(uint32_t wgID, uint32_t bucket, uint32_t v)
	{
		InterlockedAdd(statsBuff[wgID * (ConstevalParameters::AlphaBinCount + 1) + bucket], v);
	}
};
struct KernelWeightsAccessor
{
	float32_t4 get(uint32_t idx)
	{
		return kernelWeights[idx];
	}
};
struct InCSAccessor
{
	float32_t4 get(float32_t3 c, uint32_t l)
	{
		return inCS.SampleLevel(inSamp, blit::impl::dim_to_image_properties<ConstevalParameters::BlitDimCount>::getIndexCoord<float32_t>(c, l), 0);
	}
};
struct OutImgAccessor
{
	void set(int32_t3 c, uint32_t l, float32_t4 v)
	{
		outImg[blit::impl::dim_to_image_properties<ConstevalParameters::BlitDimCount>::getIndexCoord<int32_t>(c, l)] = v;
	}
};
*/

using namespace nbl::hlsl::blit;

// TODO: push constants

[numthreads(ConstevalParameters::WorkGroupSize,1,1)]
void main()
{
/*
	blit::compute_blit_t<ConstevalParameters> blit = blit::compute_blit_t<ConstevalParameters>::create(params);
    InCSAccessor inCSA;
	OutImgAccessor outImgA;
	KernelWeightsAccessor kwA;
	HistogramAccessor hA;
	SharedAccessor sA;
	blit.execute(inCSA, outImgA, kwA, hA, sA, workGroupID, localInvocationIndex);
*/
}