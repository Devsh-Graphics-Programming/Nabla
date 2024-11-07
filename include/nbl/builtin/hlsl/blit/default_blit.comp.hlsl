// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/builtin/hlsl/blit/parameters.hlsl"

#include "nbl/builtin/hlsl/blit/common.hlsl"
//#include "nbl/builtin/hlsl/blit/compute_blit.hlsl"

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

inCS.SampleLevel(inSamp, blit::impl::dim_to_image_properties<ConstevalParameters::BlitDimCount>::getIndexCoord<float32_t>(c, l), 0);
*/
struct InImgAccessor
{
	template<typename T, int32_t Dims NBL_FUNC_REQUIRES(is_same_v<T,float>)
	vector<T,4> data get(const vector<uint16_t,Dims> uv, uint16_t layer, uint16_t level)
	{
		return __get_impl<Dims>(uv,_static_cast<float>(layer),_static_cast<float>(level),data);
	}

	template<int32_t Dims>
	float32_t4 __get_impl(const vector<float32_t,Dims> uv, float layer, float level);

	uint32_t descIx : 20;
	uint32_t samplerIx : 12;
};
template<>
float32_t4 InImgAccessor::__get_impl<1>(const float32_t1 uv, float layer, float level)
{
	return inAs1DArray[descIx].SampleLevel(inSamp[samplerIx],float32_t2(uv,layer),level);
}
template<>
float32_t4 InImgAccessor::__get_impl<2>(const float32_t2 uv, float layer, float level)
{
	return inAs2DArray[descIx].SampleLevel(inSamp[samplerIx],float32_t3(uv,layer),level);
}
template<>
float32_t4 InImgAccessor::__get_impl<3>(const float32_t3 uv, float layer, float level)
{
	return inAs3D[descIx].SampleLevel(inSamp[samplerIx],uv,level);
}

using namespace nbl::hlsl::blit;

// TODO: push constants

[numthreads(ConstevalParameters::WorkGroupSize,1,1)]
void main()
{
	InImgAccessor inImgA;
	OutImgAccessor outImgA;
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