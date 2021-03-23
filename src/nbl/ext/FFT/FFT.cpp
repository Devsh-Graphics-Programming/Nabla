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

core::SRange<const SPushConstantRange> FFT::getDefaultPushConstantRanges()
{
	static const SPushConstantRange ranges[1] =
	{
		{
			ISpecializedShader::ESS_COMPUTE,
			0u,
			sizeof(Parameters_t)
		},
	};
	return {ranges, ranges+1};
}

core::smart_refctd_ptr<IGPUDescriptorSetLayout> FFT::getDefaultDescriptorSetLayout(IVideoDriver* driver)
{
	static IGPUDescriptorSetLayout::SBinding bnd[] =
	{
		{
			0u,
			EDT_STORAGE_BUFFER,
			1u,
			ISpecializedShader::ESS_COMPUTE,
			nullptr
		},
		{
			1u,
			EDT_STORAGE_BUFFER,
			1u,
			ISpecializedShader::ESS_COMPUTE,
			nullptr
		},
	};

	// TODO: cache using the asset manager's caches
	return driver->createGPUDescriptorSetLayout(bnd,bnd+sizeof(bnd)/sizeof(IGPUDescriptorSetLayout::SBinding));
}
		
//
core::smart_refctd_ptr<IGPUPipelineLayout> FFT::getDefaultPipelineLayout(IVideoDriver* driver)
{
	auto pcRange = getDefaultPushConstantRanges();
	// TODO: cache using the asset manager's caches
	return driver->createGPUPipelineLayout(
		pcRange.begin(),pcRange.end(),
		getDefaultDescriptorSetLayout(driver),nullptr,nullptr,nullptr
	);
}

core::smart_refctd_ptr<video::IGPUComputePipeline> FFT::getDefaultPipeline(video::IVideoDriver* driver, uint32_t maxDimensionSize)
{
	// TODO: cache using the asset manager's caches
	uint32_t const maxPaddedDimensionSize = core::roundUpToPoT(maxDimensionSize);

	const char* sourceFmt =
R"===(#version 430 core

#define _NBL_GLSL_WORKGROUP_SIZE_ %u
#define _NBL_GLSL_EXT_FFT_MAX_DIM_SIZE_ %u
#define _NBL_GLSL_EXT_FFT_HALF_STORAGE_ %u
 
layout(local_size_x=_NBL_GLSL_WORKGROUP_SIZE_, local_size_y=1, local_size_z=1) in;
#include "nbl/builtin/glsl/ext/FFT/default_compute_fft.comp"

)===";

	constexpr size_t extraSize = 8u*2u+1u;

	auto source = core::make_smart_refctd_ptr<ICPUBuffer>(strlen(sourceFmt)+extraSize+1u);
	snprintf(
		reinterpret_cast<char*>(source->getPointer()),source->getSize(), sourceFmt,
		DEFAULT_WORK_GROUP_SIZE,
		maxPaddedDimensionSize,
		0u
	);

	auto shader = driver->createGPUShader(core::make_smart_refctd_ptr<ICPUShader>(std::move(source),asset::ICPUShader::buffer_contains_glsl));
	
	auto specializedShader = driver->createGPUSpecializedShader(
		shader.get(),
		ISpecializedShader::SInfo{nullptr, nullptr, "main", ISpecializedShader::ESS_COMPUTE}
	);

	return driver->createGPUComputePipeline(nullptr, getDefaultPipelineLayout(driver), std::move(specializedShader));
}

void FFT::defaultBarrier()
{
	COpenGLExtensionHandler::pGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}
