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
	_NBL_STATIC_INLINE_CONSTEXPR float POSITION_EACH_AXIE_MIN = -10.f;
	_NBL_STATIC_INLINE_CONSTEXPR float POSITION_EACH_AXIE_MAX = 10.f;

	_NBL_STATIC_INLINE_CONSTEXPR float VELOCITY_EACH_AXIE_MIN = 0.f;
	_NBL_STATIC_INLINE_CONSTEXPR float VELOCITY_EACH_AXIE_MAX = 0.001f;

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

	auto* driver = device->getVideoDriver();
	auto* assetManager = device->getAssetManager();
	auto* sceneManager = device->getSceneManager();

	device->getCursorControl()->setVisible(false);

	scene::ICameraSceneNode* camera = sceneManager->addCameraSceneNodeFPS(0, 100.0f, 1.f);

	camera->setPosition(core::vector3df(0, 0, 0));
	camera->setTarget(core::vector3df(0, 0, 0));
	camera->setNearValue(0.01f);
	camera->setFarValue(100000.0f);

	sceneManager->setActiveCamera(camera);

	/*
		Compute pipeline
	*/

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
	
	auto gpuCDescriptorSetLayout = driver->createGPUDescriptorSetLayout(gpuBindingsLayout, gpuBindingsLayout + EE_COUNT);
	auto gpuCDescriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr(gpuCDescriptorSetLayout));
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
				gpuWrites[binding] = { gpuCDescriptorSet.get(), binding, 0u, 1u, EDT_STORAGE_BUFFER, gpuDescriptorSetInfos + binding };
			driver->updateDescriptorSets(EE_COUNT, gpuWrites, 0u, nullptr);
		}
	}

	auto gpuCPipelineLayout = driver->createGPUPipelineLayout(nullptr, nullptr, std::move(gpuCDescriptorSetLayout), nullptr, nullptr, nullptr);
	auto gpuComputePipeline = driver->createGPUComputePipeline(nullptr, std::move(gpuCPipelineLayout), std::move(gpuComputeShader));

	/*
		Graphics Pipeline
	*/

	asset::SVertexInputParams inputVertexParams;
	inputVertexParams.enabledAttribFlags = core::createBitmask({EE_POSITIONS, EE_VELOCITIES, EE_COLORS});
	inputVertexParams.enabledBindingFlags = core::createBitmask({ EE_POSITIONS, EE_VELOCITIES, EE_COLORS });

	for (uint8_t i = 0; i < EE_COUNT; ++i)
	{
		inputVertexParams.bindings[i].stride = getTexelOrBlockBytesize(EF_R32G32B32A32_SFLOAT);
		inputVertexParams.bindings[i].inputRate = asset::EVIR_PER_VERTEX;

		inputVertexParams.attributes[i].binding = i;
		inputVertexParams.attributes[i].format = asset::EF_R32G32B32A32_SFLOAT;
		inputVertexParams.attributes[i].relativeOffset = 0;
	}

	asset::SBlendParams blendParams;
	asset::SPrimitiveAssemblyParams primitiveAssemblyParams;
	primitiveAssemblyParams.primitiveType = EPT_POINT_LIST;
	asset::SRasterizationParams rasterizationParams;

	IGPUDescriptorSetLayout::SBinding gpuUboBinding;
	gpuUboBinding.count = 1u;
	gpuUboBinding.binding = 0;
	gpuUboBinding.stageFlags = static_cast<asset::ICPUSpecializedShader::E_SHADER_STAGE>(asset::ICPUSpecializedShader::ESS_VERTEX | asset::ICPUSpecializedShader::ESS_FRAGMENT);
	gpuUboBinding.type = asset::EDT_UNIFORM_BUFFER;

	auto gpuGDs1Layout = driver->createGPUDescriptorSetLayout(&gpuUboBinding, &gpuUboBinding + 1);
	auto gpuUBO = driver->createDeviceLocalGPUBufferOnDedMem(sizeof(SBasicViewParameters));

	auto gpuGDescriptorSet1 = driver->createGPUDescriptorSet(gpuGDs1Layout);
	{
		video::IGPUDescriptorSet::SWriteDescriptorSet write;
		write.dstSet = gpuGDescriptorSet1.get();
		write.binding = 0;
		write.count = 1u;
		write.arrayElement = 0u;
		write.descriptorType = asset::EDT_UNIFORM_BUFFER;
		video::IGPUDescriptorSet::SDescriptorInfo info;
		{
			info.desc = gpuUBO;
			info.buffer.offset = 0ull;
			info.buffer.size = sizeof(SBasicViewParameters);
		}
		write.info = &info;
		driver->updateDescriptorSets(1u, &write, 0u, nullptr);
	}

	auto vertexShaderBundle = assetManager->getAsset("../vertexShader.vert", {});
	assert(!vertexShaderBundle.isEmpty());

	auto cpuVertexShader = core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(vertexShaderBundle.getContents().begin()[0]);
	auto gpuVertexShader = driver->getGPUObjectsFromAssets(&cpuVertexShader, &cpuVertexShader + 1)->front();

	auto fragmentShaderBundle = assetManager->getAsset("../fragmentShader.frag", {});
	assert(!fragmentShaderBundle.isEmpty());

	auto cpuFragmentShader = core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(fragmentShaderBundle.getContents().begin()[0]);
	auto gpuFragmentShader = driver->getGPUObjectsFromAssets(&cpuFragmentShader, &cpuFragmentShader + 1)->front();

	core::smart_refctd_ptr<video::IGPUSpecializedShader> gpuGShaders[] = { gpuVertexShader, gpuFragmentShader };
	auto gpuGShadersPointer = reinterpret_cast<video::IGPUSpecializedShader**>(gpuGShaders);

	auto gpuGPipelineLayout = driver->createGPUPipelineLayout(nullptr, nullptr, nullptr, std::move(gpuGDs1Layout), nullptr, nullptr);
	auto gpuGraphicsPipeline = driver->createGPURenderpassIndependentPipeline(nullptr, std::move(gpuGPipelineLayout), gpuGShadersPointer, gpuGShadersPointer + sizeof(gpuGShaders) / sizeof(core::smart_refctd_ptr<video::IGPUSpecializedShader>), inputVertexParams, blendParams, primitiveAssemblyParams, rasterizationParams);

	asset::SBufferBinding<video::IGPUBuffer> gpuGbindings[video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];

	gpuGbindings[EE_POSITIONS].buffer = gpuSSBOBuffer;
	gpuGbindings[EE_POSITIONS].offset = 0;

	gpuGbindings[EE_VELOCITIES].buffer = gpuSSBOBuffer;
	gpuGbindings[EE_VELOCITIES].offset = sizeof(SShaderStorageBufferObject::positions);

	gpuGbindings[EE_COLORS].buffer = gpuSSBOBuffer;
	gpuGbindings[EE_COLORS].offset = gpuGbindings[EE_VELOCITIES].offset + sizeof(SShaderStorageBufferObject::velocities);

	auto gpuMeshBuffer = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(std::move(gpuGraphicsPipeline), nullptr, gpuGbindings, asset::SBufferBinding<video::IGPUBuffer>());
	{
		gpuMeshBuffer->setIndexType(asset::EIT_UNKNOWN);
		gpuMeshBuffer->setIndexCount(NUMBER_OF_PARTICLES);
	}

	uint64_t lastFPSTime = 0;
	while (device->run() && receiver.keepOpen())
	{
		driver->beginScene(true, true, video::SColor(0, 0, 0, 0));

		/*
			Calculation of particle postitions takes place here
		*/

		driver->bindComputePipeline(gpuComputePipeline.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, gpuComputePipeline->getLayout(), 0, 1, &gpuCDescriptorSet.get(), nullptr);

		static_assert(NUMBER_OF_PARTICLES % WORK_GROUP_SIZE == 0, "Inccorect amount!");
		_NBL_STATIC_INLINE_CONSTEXPR size_t groupCountX = NUMBER_OF_PARTICLES / WORK_GROUP_SIZE;

		driver->dispatch(groupCountX, 1, 1);
		COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		COpenGLExtensionHandler::extGlMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT | GL_PIXEL_BUFFER_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
	
		/*
			After calculation of positions each particle gets displayed
		*/

		camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
		camera->render();

		const auto viewProjection = camera->getConcatenatedMatrix();
		core::matrix3x4SIMD modelMatrix;
		modelMatrix.setTranslation(core::vectorSIMDf(0, 0, 10, 0));
		modelMatrix.setRotation(irr::core::quaternion(0, 0, 0));

		auto mv = core::concatenateBFollowedByA(camera->getViewMatrix(), modelMatrix);
		auto mvp = core::concatenateBFollowedByA(viewProjection, modelMatrix);
		core::matrix3x4SIMD normalMat;
		mv.getSub3x3InverseTranspose(normalMat);

		SBasicViewParameters uboData;
		memcpy(uboData.MV, mv.pointer(), sizeof(mv));
		memcpy(uboData.MVP, mvp.pointer(), sizeof(mvp));
		memcpy(uboData.NormalMat, normalMat.pointer(), sizeof(normalMat));
		driver->updateBufferRangeViaStagingBuffer(gpuUBO.get(), 0ull, sizeof(uboData), &uboData);

		driver->bindGraphicsPipeline(gpuMeshBuffer->getPipeline());
		driver->bindDescriptorSets(video::EPBP_GRAPHICS, gpuMeshBuffer->getPipeline()->getLayout(), 1u, 1u, &gpuGDescriptorSet1.get(), nullptr);
		driver->drawMeshBuffer(gpuMeshBuffer.get());

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
