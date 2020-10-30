// Copyright (C) 2020 - AnastaZIuk
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <irrlicht.h>
#include "../common/QToQuitEventReceiver.h"

using namespace irr;
using namespace asset;
using namespace core;
using namespace video;

_NBL_STATIC_INLINE_CONSTEXPR size_t NUMBER_OF_PARTICLES = 1024 * 1024;		// total number of particles to move
_NBL_STATIC_INLINE_CONSTEXPR size_t WORK_GROUP_SIZE = 128;					// work-items per work-group

enum E_ENTRIES
{
	EE_POSITIONS,
	EE_VELOCITIES, 
	EE_COLORS,
	EE_COUNT
};

#include "irr/irrpack.h"
struct alignas(16) SShaderStorageBufferObject
{
	core::vector4df_SIMD positions[NUMBER_OF_PARTICLES];
	core::vector4df_SIMD velocities[NUMBER_OF_PARTICLES];
	core::vector4df_SIMD colors[NUMBER_OF_PARTICLES];
} PACK_STRUCT;
#include "irr/irrunpack.h"

static_assert(sizeof(SShaderStorageBufferObject) == sizeof(SShaderStorageBufferObject::positions) + sizeof(SShaderStorageBufferObject::velocities) + sizeof(SShaderStorageBufferObject::colors), "There will be inproper alignment!");

void triggerRandomSetup(SShaderStorageBufferObject* ssbo)
{
	_NBL_STATIC_INLINE_CONSTEXPR float POSITION_EACH_AXIE_MIN = 0.f;
	_NBL_STATIC_INLINE_CONSTEXPR float POSITION_EACH_AXIE_MAX = 3.f;

	_NBL_STATIC_INLINE_CONSTEXPR float VELOCITY_EACH_AXIE_MIN = 0.f;
	_NBL_STATIC_INLINE_CONSTEXPR float VELOCITY_EACH_AXIE_MAX = 1.f;

	_NBL_STATIC_INLINE_CONSTEXPR float COLOR_EACH_AXIE_MIN = 0.f;
	_NBL_STATIC_INLINE_CONSTEXPR float COLOR_EACH_AXIE_MAX = 1.f;

	auto get_random = [&](const float& min, const float& max)
	{
		static std::default_random_engine engine;
		static std::uniform_real_distribution<> distribution(min, max);
		return distribution(engine);
	};

	for (size_t i = 0; i < NUMBER_OF_PARTICLES; ++i)
	{
		ssbo->positions[i] = core::vector4df_SIMD(get_random(POSITION_EACH_AXIE_MIN, POSITION_EACH_AXIE_MAX), get_random(POSITION_EACH_AXIE_MIN, POSITION_EACH_AXIE_MAX), get_random(POSITION_EACH_AXIE_MIN, POSITION_EACH_AXIE_MAX), get_random(POSITION_EACH_AXIE_MIN, POSITION_EACH_AXIE_MAX));
		ssbo->velocities[i] = core::vector4df_SIMD(get_random(VELOCITY_EACH_AXIE_MIN, VELOCITY_EACH_AXIE_MAX), get_random(VELOCITY_EACH_AXIE_MIN, VELOCITY_EACH_AXIE_MAX), get_random(VELOCITY_EACH_AXIE_MIN, VELOCITY_EACH_AXIE_MAX), get_random(VELOCITY_EACH_AXIE_MIN, VELOCITY_EACH_AXIE_MAX));
		ssbo->colors[i] = core::vector4df_SIMD(get_random(COLOR_EACH_AXIE_MIN, COLOR_EACH_AXIE_MAX), get_random(COLOR_EACH_AXIE_MIN, COLOR_EACH_AXIE_MAX), get_random(COLOR_EACH_AXIE_MIN, COLOR_EACH_AXIE_MAX), get_random(COLOR_EACH_AXIE_MIN, COLOR_EACH_AXIE_MAX));
	}	
}

int main()
{
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; 
	params.ZBufferBits = 24; 
	params.DriverType = video::EDT_OPENGL; 
	params.WindowSize = dimension2d<uint32_t>(1280, 720);
	params.Fullscreen = false;
	params.Vsync = true;
	params.Doublebuffer = true;
	params.Stencilbuffer = false; 
	auto device = createDeviceEx(params);

	if (!device)
		return 1;

	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);

	auto driver = device->getVideoDriver();
	auto assetManager = device->getAssetManager();

	auto computeShaderBundle = assetManager->getAsset("../computeShader.comp", {});
	assert(!computeShaderBundle.isEmpty());

	auto cpuComputeShader = core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(computeShaderBundle.getContents().begin()[0]);
	auto gpuComputeShader = driver->getGPUObjectsFromAssets(&cpuComputeShader, &cpuComputeShader + 1)->front();

	auto cpuSSBOBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(sizeof(SShaderStorageBufferObject));
	triggerRandomSetup(reinterpret_cast<SShaderStorageBufferObject*>(cpuSSBOBuffer->getPointer()));
	auto gpuSSBOOffsetBufferPair = driver->getGPUObjectsFromAssets(&cpuSSBOBuffer, &cpuSSBOBuffer + 1)->front();
	auto gpuSSBOBuffer = core::smart_refctd_ptr<IGPUBuffer>(gpuSSBOOffsetBufferPair->getBuffer());

	IGPUDescriptorSetLayout::SBinding gpuBindingsLayout[EE_COUNT] =
	{
		{EE_POSITIONS, EDT_STORAGE_BUFFER, 1u, IGPUSpecializedShader::ESS_COMPUTE, nullptr},
		{EE_VELOCITIES, EDT_STORAGE_BUFFER, 1u, IGPUSpecializedShader::ESS_COMPUTE, nullptr},
		{EE_COLORS, EDT_STORAGE_BUFFER, 1u, IGPUSpecializedShader::ESS_COMPUTE, nullptr}
	};
	
	auto gpuDescriptorSetLayout = driver->createGPUDescriptorSetLayout(gpuBindingsLayout, gpuBindingsLayout + EE_COUNT);
	auto gpuDescriptorSetCompute = driver->createGPUDescriptorSet(core::smart_refctd_ptr(gpuDescriptorSetLayout));
	{
		IGPUDescriptorSet::SDescriptorInfo gpuDescriptorSetInfos[EE_COUNT];

		gpuDescriptorSetInfos[EE_POSITIONS].desc = gpuSSBOBuffer;
		gpuDescriptorSetInfos[EE_POSITIONS].buffer.size = sizeof(SShaderStorageBufferObject::positions);
		gpuDescriptorSetInfos[EE_POSITIONS].buffer.offset = 0;

		gpuDescriptorSetInfos[EE_VELOCITIES].desc = gpuSSBOBuffer;
		gpuDescriptorSetInfos[EE_VELOCITIES].buffer.size = sizeof(SShaderStorageBufferObject::velocities);
		gpuDescriptorSetInfos[EE_VELOCITIES].buffer.offset = sizeof(SShaderStorageBufferObject::positions);

		gpuDescriptorSetInfos[EE_COLORS].desc = gpuSSBOBuffer;
		gpuDescriptorSetInfos[EE_COLORS].buffer.size = sizeof(SShaderStorageBufferObject::colors);
		gpuDescriptorSetInfos[EE_COLORS].buffer.offset = gpuDescriptorSetInfos[EE_VELOCITIES].buffer.offset + sizeof(SShaderStorageBufferObject::velocities);

		IGPUDescriptorSet::SWriteDescriptorSet gpuWrites[EE_COUNT];
		{
			for (uint32_t binding = 0u; binding < EE_COUNT; binding++)
				gpuWrites[binding] = { gpuDescriptorSetCompute.get(), binding, 0u, 1u, EDT_STORAGE_BUFFER, gpuDescriptorSetInfos + binding };
			driver->updateDescriptorSets(EE_COUNT, gpuWrites, 0u, nullptr);
		}
	}

	auto gpuPipelineLayout = driver->createGPUPipelineLayout(nullptr, nullptr, std::move(gpuDescriptorSetLayout), nullptr, nullptr, nullptr);
	auto gpuComputePipeline = driver->createGPUComputePipeline(nullptr, std::move(gpuPipelineLayout), std::move(gpuComputeShader));

	uint64_t lastFPSTime = 0;
	while (device->run() && receiver.keepOpen())
	{
		driver->beginScene(true, true, video::SColor(255, 255, 255, 255));

		driver->bindComputePipeline(gpuComputePipeline.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, gpuComputePipeline->getLayout(), 1, 1, &gpuDescriptorSetCompute.get(), nullptr);
		driver->dispatch(NUMBER_OF_PARTICLES / WORK_GROUP_SIZE, 1, 1);
		COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		COpenGLExtensionHandler::extGlMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT | GL_PIXEL_BUFFER_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
	
		// TODO - graphics pipeline! Use new positions to display points

		driver->endScene();

		uint64_t time = device->getTimer()->getRealTime();
		if (time - lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"Compute Shaders - Nabla Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str().c_str());
			lastFPSTime = time;
		}
	}
}
