// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BLIT_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_BLIT_COMMON_INCLUDED_

#include <nbl/builtin/hlsl/binding_info.hlsl>

#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
namespace nbl
{
namespace hlsl
{
namespace glsl
{
uint32_t3 gl_WorkGroupSize()
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

[[vk::push_constant]] const nbl::hlsl::blit::Parameters pc;


#include <nbl/builtin/hlsl/concepts.hlsl>
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
*/

struct OutImgAccessor
{
	template<typename T, int32_t Dims NBL_FUNC_REQUIRES(is_same_v<T,float>)
	void set(const vector<uint16_t,Dims> uv, uint16_t layer, const vector<T,4> data)
	{
		return __set_impl<Dims>(uv,layer,data);
	}

	template<int32_t Dims>
	void __set_impl(const vector<uint16_t,Dims> uv, uint16_t layer, const float32_t4 data);

	uint32_t descIx;
};
template<>
void OutImgAccessor::__set_impl<1>(const uint16_t1 uv, uint16_t layer, const float32_t4 data)
{
	outAs1DArray[descIx][uint32_t2(uv,layer)] = data;
}
template<>
void OutImgAccessor::__set_impl<2>(const uint16_t2 uv, uint16_t layer, const float32_t4 data)
{
	outAs2DArray[descIx][uint32_t3(uv,layer)] = data;
}
template<>
void OutImgAccessor::__set_impl<3>(const uint16_t3 uv, uint16_t layer, const float32_t4 data)
{
	outAs3D[descIx][uv] = data;
}
#endif