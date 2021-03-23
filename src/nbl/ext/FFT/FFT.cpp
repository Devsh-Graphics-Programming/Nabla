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

FFT::FFT(IDriver* driver, uint32_t maxDimensionSize, bool useHalfStorage) : m_maxFFTLen(core::roundUpToPoT(maxDimensionSize)), m_halfFloatStorage(useHalfStorage)
{
	// TODO: cache layouts using asset mgr or something
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
	m_dsLayout = driver->createGPUDescriptorSetLayout(bnd,bnd+sizeof(bnd)/sizeof(IGPUDescriptorSetLayout::SBinding));

	auto pcRange = getDefaultPushConstantRanges();
	m_pplnLayout = driver->createGPUPipelineLayout(pcRange.begin(),pcRange.end(),core::smart_refctd_ptr(m_dsLayout));

	if (m_maxFFTLen < MINIMUM_FFT_SIZE)
		m_maxFFTLen = MINIMUM_FFT_SIZE;

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
		m_maxFFTLen,
		useHalfStorage ? 1u:0u
	);

	auto shader = driver->createGPUShader(core::make_smart_refctd_ptr<ICPUShader>(std::move(source),asset::ICPUShader::buffer_contains_glsl));
	
	auto specializedShader = driver->createGPUSpecializedShader(
		shader.get(),
		ISpecializedShader::SInfo{nullptr, nullptr, "main", ISpecializedShader::ESS_COMPUTE}
	);

	m_ppln = driver->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(m_pplnLayout),std::move(specializedShader));
}

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
	return {ranges,ranges+1};
}

void FFT::updateDescriptorSet(
	video::IVideoDriver * driver,
	video::IGPUDescriptorSet * set,
	core::smart_refctd_ptr<video::IGPUBuffer> inputBufferDescriptor,
	core::smart_refctd_ptr<video::IGPUBuffer> outputBufferDescriptor)
{
	constexpr uint32_t MAX_DESCRIPTOR_COUNT = 2u;
	video::IGPUDescriptorSet::SDescriptorInfo pInfos[MAX_DESCRIPTOR_COUNT];
	video::IGPUDescriptorSet::SWriteDescriptorSet pWrites[MAX_DESCRIPTOR_COUNT];

	for (auto i=0; i< MAX_DESCRIPTOR_COUNT; i++)
	{
		pWrites[i].dstSet = set;
		pWrites[i].arrayElement = 0u;
		pWrites[i].count = 1u;
		pWrites[i].info = pInfos+i;
	}

	// Input Buffer 
	pWrites[0].binding = 0;
	pWrites[0].descriptorType = asset::EDT_STORAGE_BUFFER;
	pWrites[0].count = 1;
	pInfos[0].desc = inputBufferDescriptor;
	pInfos[0].buffer.size = inputBufferDescriptor->getSize();
	pInfos[0].buffer.offset = 0u;

	// Output Buffer 
	pWrites[1].binding = 1;
	pWrites[1].descriptorType = asset::EDT_STORAGE_BUFFER;
	pWrites[1].count = 1;
	pInfos[1].desc = outputBufferDescriptor;
	pInfos[1].buffer.size = outputBufferDescriptor->getSize();
	pInfos[1].buffer.offset = 0u;

	driver->updateDescriptorSets(2u, pWrites, 0u, nullptr);
}

void FFT::defaultBarrier()
{
	COpenGLExtensionHandler::pGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}
