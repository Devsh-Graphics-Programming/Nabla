// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/FFT/FFT.h"
#include "../../../../source/Nabla/COpenGLExtensionHandler.h"

#include <cstdio>

using namespace nbl;
using namespace nbl::asset;
using namespace nbl::video;
using namespace ext::FFT;

core::SRange<const asset::SPushConstantRange> FFT::getDefaultPushConstantRanges()
{
	static const asset::SPushConstantRange ranges[5] =
	{
		{
			ISpecializedShader::ESS_COMPUTE,
			0u,
			sizeof(uint32_t) * 3
		},
		{
			ISpecializedShader::ESS_COMPUTE,
			sizeof(uint32_t) * 4,
			sizeof(uint32_t) * 3
		},
		{
			ISpecializedShader::ESS_COMPUTE,
			sizeof(uint32_t) * 8,
			sizeof(uint32_t)
		},
		{
			ISpecializedShader::ESS_COMPUTE,
			sizeof(uint32_t) * 9,
			sizeof(uint32_t)
		},
		{
			ISpecializedShader::ESS_COMPUTE,
			sizeof(uint32_t) * 10,
			sizeof(uint32_t)
		},
	};
	return {ranges, ranges+5};
}

core::SRange<const video::IGPUDescriptorSetLayout::SBinding> FFT::getDefaultBindings(video::IVideoDriver* driver, DataType inputType)
{
	static core::smart_refctd_ptr<IGPUSampler> sampler;

	static IGPUDescriptorSetLayout::SBinding bnd[] =
	{
		{
			0u,
			EDT_STORAGE_BUFFER,
			1u,
			ISpecializedShader::ESS_COMPUTE,
			&sampler
		},
		{
			1u,
			EDT_STORAGE_BUFFER,
			1u,
			ISpecializedShader::ESS_COMPUTE,
			nullptr
		},
	};

	if (DataType::SSBO == inputType) {
		bnd[0].type = EDT_STORAGE_BUFFER;
	} else if (DataType::TEXTURE2D == inputType) {
		bnd[0].type = EDT_COMBINED_IMAGE_SAMPLER;
	}

	bnd[0].samplers = &sampler;
	
	if (!sampler)
	{
		IGPUSampler::SParams params =
		{
			{
				ISampler::ETC_CLAMP_TO_EDGE,
				ISampler::ETC_CLAMP_TO_EDGE,
				ISampler::ETC_CLAMP_TO_EDGE,
				ISampler::ETBC_FLOAT_OPAQUE_BLACK,
				ISampler::ETF_NEAREST,
				ISampler::ETF_NEAREST,
				ISampler::ESMM_NEAREST,
				0u,
				0u,
				ISampler::ECO_ALWAYS
			}
		};
		sampler = driver->createGPUSampler(params);
	}

	return {bnd, bnd+sizeof(bnd)/sizeof(IGPUDescriptorSetLayout::SBinding)};
}

core::smart_refctd_ptr<video::IGPUSpecializedShader> FFT::createShader(video::IVideoDriver* driver, DataType inputType, uint32_t maxPaddedDimensionSize)
{
	assert(core::isPoT(maxPaddedDimensionSize));

	const char* sourceFmt =
R"===(#version 430 core

#define USE_SSBO_FOR_INPUT %u

// WorkGroup Size

#ifndef _NBL_GLSL_EXT_FFT_BLOCK_SIZE_X_DEFINED_
#define _NBL_GLSL_EXT_FFT_BLOCK_SIZE_X_DEFINED_ %u
#endif

#ifndef _NBL_GLSL_EXT_FFT_BLOCK_SIZE_Y_DEFINED_
#define _NBL_GLSL_EXT_FFT_BLOCK_SIZE_Y_DEFINED_ 1
#endif

#ifndef _NBL_GLSL_EXT_FFT_BLOCK_SIZE_Z_DEFINED_
#define _NBL_GLSL_EXT_FFT_BLOCK_SIZE_Z_DEFINED_ 1
#endif

#define _NBL_GLSL_WORKGROUP_SIZE_ (_NBL_GLSL_EXT_FFT_BLOCK_SIZE_X_DEFINED_*_NBL_GLSL_EXT_FFT_BLOCK_SIZE_Y_DEFINED_*_NBL_GLSL_EXT_FFT_BLOCK_SIZE_Z_DEFINED_)

layout(local_size_x=_NBL_GLSL_EXT_FFT_BLOCK_SIZE_X_DEFINED_, local_size_y=_NBL_GLSL_EXT_FFT_BLOCK_SIZE_Y_DEFINED_, local_size_z=_NBL_GLSL_EXT_FFT_BLOCK_SIZE_Z_DEFINED_) in;
 
#define _NBL_GLSL_EXT_FFT_MAX_DIM_SIZE_ %u
#define _NBL_GLSL_EXT_FFT_MAX_ITEMS_PER_THREAD %u
 
#define _NBL_GLSL_EXT_FFT_GET_DATA_DEFINED_
#define _NBL_GLSL_EXT_FFT_SET_DATA_DEFINED_
#include "nbl/builtin/glsl/ext/FFT/fft.glsl"

// Input Descriptor

struct nbl_glsl_ext_FFT_input_t
{
	vec2 complex_value;
};

#ifndef _NBL_GLSL_EXT_FFT_INPUT_SET_DEFINED_
#define _NBL_GLSL_EXT_FFT_INPUT_SET_DEFINED_ 0
#endif

#ifndef _NBL_GLSL_EXT_FFT_INPUT_BINDING_DEFINED_
#define _NBL_GLSL_EXT_FFT_INPUT_BINDING_DEFINED_ 0
#endif

#ifndef _NBL_GLSL_EXT_FFT_INPUT_DESCRIPTOR_DEFINED_

#if USE_SSBO_FOR_INPUT > 0
#define _NBL_GLSL_EXT_FFT_INPUT_DESCRIPTOR_DEFINED_
layout(set=_NBL_GLSL_EXT_FFT_INPUT_SET_DEFINED_, binding=_NBL_GLSL_EXT_FFT_INPUT_BINDING_DEFINED_) readonly restrict buffer InputBuffer
{
	nbl_glsl_ext_FFT_input_t inData[];
};
#else 
layout(set=_NBL_GLSL_EXT_FFT_INPUT_SET_DEFINED_, binding=_NBL_GLSL_EXT_FFT_INPUT_BINDING_DEFINED_) uniform sampler2D inputImage;
#endif

#endif

// Output Descriptor

struct nbl_glsl_ext_FFT_output_t
{
	vec2 complex_value;
};


#ifndef _NBL_GLSL_EXT_FFT_OUTPUT_SET_DEFINED_
#define _NBL_GLSL_EXT_FFT_OUTPUT_SET_DEFINED_ 0
#endif

#ifndef _NBL_GLSL_EXT_FFT_OUTPUT_BINDING_DEFINED_
#define _NBL_GLSL_EXT_FFT_OUTPUT_BINDING_DEFINED_ 1
#endif

#ifndef _NBL_GLSL_EXT_FFT_OUTPUT_DESCRIPTOR_DEFINED_
#define _NBL_GLSL_EXT_FFT_OUTPUT_DESCRIPTOR_DEFINED_
layout(set=_NBL_GLSL_EXT_FFT_OUTPUT_SET_DEFINED_, binding=_NBL_GLSL_EXT_FFT_OUTPUT_BINDING_DEFINED_) restrict buffer OutputBuffer
{
	nbl_glsl_ext_FFT_output_t outData[];
};
#endif

// Get/Set Data Function

vec2 nbl_glsl_ext_FFT_getData(in uvec3 coordinate, in uint channel)
{
	vec2 retValue = vec2(0, 0);
#if USE_SSBO_FOR_INPUT > 0
	uvec3 dimension = pc.dimension;
	uint index = channel * (dimension.x * dimension.y * dimension.z) + coordinate.z * (dimension.x * dimension.y) + coordinate.y * (dimension.x) + coordinate.x;
	retValue = inData[index].complex_value;
#else
	vec4 texelValue= texelFetch(inputImage, ivec2(coordinate.xy), 0);
	retValue = vec2(texelValue[channel], 0.0f);
#endif
	return retValue;
}

void nbl_glsl_ext_FFT_setData(in uvec3 coordinate, in uint channel, in vec2 complex_value)
{
	uvec3 dimension = pc.padded_dimension;
	uint index = channel * (dimension.x * dimension.y * dimension.z) + coordinate.z * (dimension.x * dimension.y) + coordinate.y * (dimension.x) + coordinate.x;
	outData[index].complex_value = complex_value;
}

void main()
{
	nbl_glsl_ext_FFT();
}

)===";

	const size_t extraSize = 32 + 32 + 32 + 32;

	const uint32_t maxItemsPerThread = core::ceil(float(maxPaddedDimensionSize) / DEFAULT_WORK_GROUP_X_DIM);
	const uint32_t useSSBO = (DataType::SSBO == inputType) ? 1 : 0;
	auto shader = core::make_smart_refctd_ptr<ICPUBuffer>(strlen(sourceFmt)+extraSize+1u);
	snprintf(
		reinterpret_cast<char*>(shader->getPointer()),shader->getSize(), sourceFmt,
		useSSBO,
		DEFAULT_WORK_GROUP_X_DIM,
		maxPaddedDimensionSize,
		maxItemsPerThread
	);

	auto cpuSpecializedShader = core::make_smart_refctd_ptr<ICPUSpecializedShader>(
		core::make_smart_refctd_ptr<ICPUShader>(std::move(shader),ICPUShader::buffer_contains_glsl),
		ISpecializedShader::SInfo{nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE}
	);
	
	auto gpuShader = driver->createGPUShader(nbl::core::smart_refctd_ptr<const ICPUShader>(cpuSpecializedShader->getUnspecialized()));
	
	auto gpuSpecializedShader = driver->createGPUSpecializedShader(gpuShader.get(), cpuSpecializedShader->getSpecializationInfo());

	return gpuSpecializedShader;
}

void FFT::defaultBarrier()
{
	COpenGLExtensionHandler::pGlMemoryBarrier(GL_UNIFORM_BARRIER_BIT|GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);
}