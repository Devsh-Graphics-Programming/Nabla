// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/builtin/hlsl/blit/parameters.hlsl"
#include "nbl/builtin/hlsl/blit/compute_blit.hlsl"

#include "nbl/builtin/hlsl/blit/common.hlsl"

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
*/

namespace nbl
{
namespace hlsl
{
template<typename T>
struct SImageDimensions
{
/*
	static SImageDimensions query(NBL_CONST_REF_ARG(Texture1DArray) tex)
	{
		SImageDimensions image;
		return 
	}
*/

	T width,height,depth;
	uint16_t layers;
	uint16_t levels : 5;
	uint16_t samples : 6;
};
}
}

struct InImgAccessor
{
	template<int32_t Dims>
	vector<float32_t,Dims> extentRcp(const uint16_t level)
	{
		return truncate<Dims>(pc.inputImageExtentRcp);
	}

	template<typename T, int32_t Dims NBL_FUNC_REQUIRES(is_same_v<T,float>)
	vector<T,4> get(const vector<float32_t,Dims> uv, uint16_t layer, uint16_t level)
	{
		return __get_impl<Dims>(uv,_static_cast<float>(layer),_static_cast<float>(level));
	}
	
// private
	template<int32_t Dims>
	float32_t4 __get_impl(const vector<float32_t,Dims> uv, float layer, float level);

	uint32_t descIx;
	uint32_t samplerIx;
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

// https://github.com/microsoft/DirectXShaderCompiler/issues/7001
//[numthreads(ConstevalParameters::WorkGroupSize,1,1)]
[numthreads(NBL_WORKGROUP_SIZE,1,1)]
void main()
{
	InImgAccessor inImgA;
	inImgA.descIx = pc.inputDescIx;
	inImgA.samplerIx = pc.samplerDescIx;

	OutImgAccessor outImgA;
	outImgA.descIx = pc.outputDescIx;

	const uint16_t3 virtWorkGroupID = _static_cast<uint16_t3>(glsl::gl_WorkGroupID());
	const uint16_t layer = virtWorkGroupID.z;
	// TODO: If and when someone can be bothered, change the blit api to compile a pipeline per image dimension, maybe it will be faster. Library target could be useful for that!
	switch (pc.perWG.imageDim)
	{
		case 1:
			if (pc.perWG.doCoverage())
				blit::execute<true,ConstevalParameters::WorkGroupSize>(inImgA,outImgA,/*kernelW,histoA,*/sharedAccessor,pc.perWG,layer,uint16_t1(virtWorkGroupID.x));
			else
				blit::execute<false,ConstevalParameters::WorkGroupSize>(inImgA,outImgA,/*kernelW,histoA,*/sharedAccessor,pc.perWG,layer,uint16_t1(virtWorkGroupID.x));
			break;
		case 2:
			if (pc.perWG.doCoverage())
				blit::execute<true,ConstevalParameters::WorkGroupSize>(inImgA,outImgA,/*kernelW,histoA,*/sharedAccessor,pc.perWG,layer,virtWorkGroupID.xy);
			else
				blit::execute<false,ConstevalParameters::WorkGroupSize>(inImgA,outImgA,/*kernelW,histoA,*/sharedAccessor,pc.perWG,layer,virtWorkGroupID.xy);
			break;
		case 3:
			if (pc.perWG.doCoverage())
				blit::execute<true,ConstevalParameters::WorkGroupSize>(inImgA,outImgA,/*kernelW,histoA,*/sharedAccessor,pc.perWG,layer,virtWorkGroupID);
			else
				blit::execute<false,ConstevalParameters::WorkGroupSize>(inImgA,outImgA,/*kernelW,histoA,*/sharedAccessor,pc.perWG,layer,virtWorkGroupID);
			break;
	}
}