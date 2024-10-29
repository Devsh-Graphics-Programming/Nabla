// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BLIT_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_BLIT_COMMON_INCLUDED_

#include <nbl/builtin/hlsl/binding_info.hlsl>

namespace nbl
{
namespace hlsl
{
namespace glsl
{
uint32_t gl_WorkGroupSize()
{
	return uint32_t3(ConstevalParameters::WorkGroupSize,1,1);
}
}
}
}

using namespace nbl::hlsl;

[[vk::binding(ConstevalParameters::kernel_weight_binding_t::Index,ConstevalParameters::kernel_weight_binding_t::Set)]]
Buffer<float32_t4> kernelWeights[ConstevalParameters::kernel_weight_binding_t::Count];
[[vk::binding(ConstevalParameters::input_sampler_binding_t::Index,ConstevalParameters::input_sampler_binding_t::Set)]]
SamplerState inSamp[ConstevalParameters::input_sampler_binding_t::Count];
// aliased
[[vk::binding(ConstevalParameters::input_image_binding_t::Index,ConstevalParameters::input_image_binding_t::Set)]]
Texture1DArray<float4> inAs1DArray[ConstevalParameters::input_image_binding_t::Count];
[[vk::binding(ConstevalParameters::input_image_binding_t::Index,ConstevalParameters::input_image_binding_t::Set)]]
Texture2DArray<float4> inAs2DArray[ConstevalParameters::input_image_binding_t::Count];
[[vk::binding(ConstevalParameters::input_image_binding_t::Index,ConstevalParameters::input_image_binding_t::Set)]]
Texture3D<float4> inAs3D[ConstevalParameters::input_image_binding_t::Count];
// aliased
[[vk::binding(ConstevalParameters::output_binding_t::Index,ConstevalParameters::output_binding_t::Set)]] [[vk::image_format("unknown")]]
RWTexture1DArray<float4> outAs1DArray[ConstevalParameters::output_binding_t::Count];
[[vk::binding(ConstevalParameters::output_binding_t::Index,ConstevalParameters::output_binding_t::Set)]] [[vk::image_format("unknown")]]
RWTexture2DArray<float4> outAs2DArray[ConstevalParameters::output_binding_t::Count];
[[vk::binding(ConstevalParameters::output_binding_t::Index,ConstevalParameters::output_binding_t::Set)]] [[vk::image_format("unknown")]]
RWTexture3D<float4> outAs3D[ConstevalParameters::output_binding_t::Count];


groupshared uint32_t sMem[ConstevalParameters::SharedMemoryDWORDs];
/*
struct HistogramAccessor
{
	void atomicAdd(uint32_t wgID, uint32_t bucket, uint32_t v)
	{
		InterlockedAdd(statsBuff[wgID * (ConstevalParameters::AlphaBinCount + 1) + bucket], v);
	}
};
struct SharedAccessor
{
	float32_t get(float32_t idx)
	{
		return sMem[idx];
	}
	void set(float32_t idx, float32_t val)
	{
		sMem[idx] = val;
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
#endif