// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>

#include "../common/Camera.hpp"
#include "../common/CommonAPI.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"

using namespace nbl;
using namespace asset;
using namespace core;

/*
	Uncomment for more detailed logging
*/

// #define NBL_MORE_LOGS

class CEventReceiver
{
public:
	CEventReceiver() : particlesVectorChangeFlag(false), forceChangeVelocityFlag(false), visualizeVelocityVectorsFlag(false) {}

	void process(const IKeyboardEventChannel::range_t& events)
	{
		particlesVectorChangeFlag = false;
		forceChangeVelocityFlag = false;
		visualizeVelocityVectorsFlag = false;

		for (auto eventIterator = events.begin(); eventIterator != events.end(); eventIterator++)
		{
			auto event = *eventIterator;

			if (event.keyCode == nbl::ui::EKC_X)
				particlesVectorChangeFlag = true;
			
			if (event.keyCode == nbl::ui::EKC_Z)
				forceChangeVelocityFlag = true;

			if (event.keyCode == nbl::ui::EKC_C)
				visualizeVelocityVectorsFlag = true;

			if (event.keyCode == nbl::ui::EKC_V)
				visualizeVelocityVectorsFlag = false;
		}
	}

	inline bool isXPressed() const { return particlesVectorChangeFlag; }
	inline bool isZPressed() const{ return forceChangeVelocityFlag; }
	inline bool isCPressed() const { return visualizeVelocityVectorsFlag; }

private:
	bool particlesVectorChangeFlag;
	bool forceChangeVelocityFlag;
	bool visualizeVelocityVectorsFlag;
};

_NBL_STATIC_INLINE_CONSTEXPR size_t NUMBER_OF_PARTICLES = 1024 * 1024;		// total number of particles to move
_NBL_STATIC_INLINE_CONSTEXPR size_t WORK_GROUP_SIZE = 128;					// work-items per work-group

enum E_ENTRIES
{
	EE_POSITIONS,
	EE_VELOCITIES, 
	EE_COLORS,
	EE_COLORS_RISING_FLAG,
	EE_COUNT
};

#include "nbl/nblpack.h"
struct alignas(16) SShaderStorageBufferObject
{
	core::vector4df_SIMD positions[NUMBER_OF_PARTICLES];
	core::vector4df_SIMD velocities[NUMBER_OF_PARTICLES];
	core::vector4df_SIMD colors[NUMBER_OF_PARTICLES];
	bool isColorIntensityRising[NUMBER_OF_PARTICLES][4];
} PACK_STRUCT;
#include "nbl/nblunpack.h"

static_assert(sizeof(SShaderStorageBufferObject) == sizeof(SShaderStorageBufferObject::positions) + sizeof(SShaderStorageBufferObject::velocities) + sizeof(SShaderStorageBufferObject::colors) + sizeof(SShaderStorageBufferObject::isColorIntensityRising), "There will be inproper alignment!");

#include "nbl/nblpack.h"
struct alignas(32) SPushConstants
{
	uint32_t isXPressed = false;
	uint32_t isZPressed = false;
	uint32_t isCPressed = false;
	core::vector3df currentUserAbsolutePosition;
} PACK_STRUCT;
#include "nbl/nblunpack.h"

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

		for (uint8_t b = 0; b < 4; ++b)
			ssbo->isColorIntensityRising[i][b] = true;
	}	
}

int main()
{
	constexpr uint32_t WIN_W = 1280;
	constexpr uint32_t WIN_H = 720;
	constexpr uint32_t FBO_COUNT = 1u;

	auto initOutput = CommonAPI::Init<WIN_W, WIN_H, FBO_COUNT>(video::EAT_OPENGL, "MeshLoaders", nbl::asset::EF_D32_SFLOAT);
	auto window = std::move(initOutput.window);
	auto gl = std::move(initOutput.apiConnection);
	auto surface = std::move(initOutput.surface);
	auto gpuPhysicalDevice = std::move(initOutput.physicalDevice);
	auto logicalDevice = std::move(initOutput.logicalDevice);
	auto queues = std::move(initOutput.queues);
	auto swapchain = std::move(initOutput.swapchain);
	auto renderpass = std::move(initOutput.renderpass);
	auto fbo = std::move(initOutput.fbo[0]);
	auto commandPool = std::move(initOutput.commandPool);
	auto assetManager = std::move(initOutput.assetManager);
	auto logger = std::move(initOutput.logger);
	auto inputSystem = std::move(initOutput.inputSystem);
	auto windowCallback = std::move(initOutput.windowCb);
	auto cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
	nbl::video::IGPUObjectFromAssetConverter cpu2gpu;

	core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> commandBuffers[1];
	logicalDevice->createCommandBuffers(commandPool.get(), nbl::video::IGPUCommandBuffer::EL_PRIMARY, 1, commandBuffers);
	auto commandBuffer = commandBuffers[0];

	auto createDescriptorPool = [&](const uint32_t itemCount, E_DESCRIPTOR_TYPE descriptorType)
	{
		constexpr uint32_t maxItemCount = 256u;
		{
			nbl::video::IDescriptorPool::SDescriptorPoolSize poolSize;
			poolSize.count = itemCount;
			poolSize.type = descriptorType;
			return logicalDevice->createDescriptorPool(static_cast<nbl::video::IDescriptorPool::E_CREATE_FLAGS>(0), maxItemCount, 1u, &poolSize);
		}
	};

	/*
		Compute pipeline
	*/

	auto computeShaderBundle = assetManager->getAsset("../computeShader.comp", {});
	{
		bool status = !computeShaderBundle.getContents().empty();
		assert(status);
	}
	

	auto cpuComputeShader = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(computeShaderBundle.getContents().begin()[0]);
	smart_refctd_ptr<video::IGPUSpecializedShader> gpuComputeShader;
	{
		auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&cpuComputeShader, &cpuComputeShader + 1, cpu2gpuParams);
		if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
			assert(false);

		gpuComputeShader = (*gpu_array)[0];
	}

	auto cpuSSBOBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(sizeof(SShaderStorageBufferObject));
	triggerRandomSetup(reinterpret_cast<SShaderStorageBufferObject*>(cpuSSBOBuffer->getPointer()));
	core::smart_refctd_ptr<video::IGPUBuffer> gpuSSBOBuffer;
	{
		auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&cpuSSBOBuffer, &cpuSSBOBuffer + 1, cpu2gpuParams);
		if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
			assert(false);

		auto gpuSSBOOffsetBufferPair = (*gpu_array)[0];
	}

	video::IGPUDescriptorSetLayout::SBinding gpuBindingsLayout[EE_COUNT] =
	{
		{EE_POSITIONS, EDT_STORAGE_BUFFER, 1u, video::IGPUSpecializedShader::ESS_COMPUTE, nullptr},
		{EE_VELOCITIES, EDT_STORAGE_BUFFER, 1u, video::IGPUSpecializedShader::ESS_COMPUTE, nullptr},
		{EE_COLORS, EDT_STORAGE_BUFFER, 1u, video::IGPUSpecializedShader::ESS_COMPUTE, nullptr},
		{EE_COLORS_RISING_FLAG, EDT_STORAGE_BUFFER, 1u, video::IGPUSpecializedShader::ESS_COMPUTE, nullptr}
	};
	
	auto gpuCDescriptorPool = createDescriptorPool(EE_COUNT, EDT_STORAGE_BUFFER);
	auto gpuCDescriptorSetLayout = logicalDevice->createGPUDescriptorSetLayout(gpuBindingsLayout, gpuBindingsLayout + EE_COUNT);
	auto gpuCDescriptorSet = logicalDevice->createGPUDescriptorSet(gpuCDescriptorPool.get(), core::smart_refctd_ptr(gpuCDescriptorSetLayout));
	{
		video::IGPUDescriptorSet::SDescriptorInfo gpuDescriptorSetInfos[EE_COUNT];

		gpuDescriptorSetInfos[EE_POSITIONS].desc = gpuSSBOBuffer;
		gpuDescriptorSetInfos[EE_POSITIONS].buffer.size = sizeof(SShaderStorageBufferObject::positions);
		gpuDescriptorSetInfos[EE_POSITIONS].buffer.offset = 0;

		gpuDescriptorSetInfos[EE_VELOCITIES].desc = gpuSSBOBuffer;
		gpuDescriptorSetInfos[EE_VELOCITIES].buffer.size = sizeof(SShaderStorageBufferObject::velocities);
		gpuDescriptorSetInfos[EE_VELOCITIES].buffer.offset = sizeof(SShaderStorageBufferObject::positions);

		gpuDescriptorSetInfos[EE_COLORS].desc = gpuSSBOBuffer;
		gpuDescriptorSetInfos[EE_COLORS].buffer.size = sizeof(SShaderStorageBufferObject::colors);
		gpuDescriptorSetInfos[EE_COLORS].buffer.offset = gpuDescriptorSetInfos[EE_VELOCITIES].buffer.offset + sizeof(SShaderStorageBufferObject::velocities);

		gpuDescriptorSetInfos[EE_COLORS_RISING_FLAG].desc = gpuSSBOBuffer;
		gpuDescriptorSetInfos[EE_COLORS_RISING_FLAG].buffer.size = sizeof(SShaderStorageBufferObject::isColorIntensityRising);
		gpuDescriptorSetInfos[EE_COLORS_RISING_FLAG].buffer.offset = gpuDescriptorSetInfos[EE_COLORS].buffer.offset + sizeof(SShaderStorageBufferObject::colors);

		video::IGPUDescriptorSet::SWriteDescriptorSet gpuWrites[EE_COUNT];
		{
			for (uint32_t binding = 0u; binding < EE_COUNT; binding++)
				gpuWrites[binding] = { gpuCDescriptorSet.get(), binding, 0u, 1u, EDT_STORAGE_BUFFER, gpuDescriptorSetInfos + binding };
			logicalDevice->updateDescriptorSets(EE_COUNT, gpuWrites, 0u, nullptr);
		}
	}

	asset::SPushConstantRange pushConstantRange;
	{
		pushConstantRange.stageFlags = (asset::ISpecializedShader::E_SHADER_STAGE)(asset::ISpecializedShader::ESS_COMPUTE | asset::ISpecializedShader::ESS_GEOMETRY);
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(SPushConstants);
	}

	auto gpuCPipelineLayout = logicalDevice->createGPUPipelineLayout(&pushConstantRange, &pushConstantRange + 1, std::move(gpuCDescriptorSetLayout), nullptr, nullptr, nullptr);
	auto gpuComputePipeline = logicalDevice->createGPUComputePipeline(nullptr, std::move(gpuCPipelineLayout), std::move(gpuComputeShader));

	/*
		Graphics Pipeline
	*/

	asset::SVertexInputParams inputVertexParams;
	inputVertexParams.enabledAttribFlags = core::createBitmask({EE_POSITIONS, EE_VELOCITIES, EE_COLORS, EE_COLORS_RISING_FLAG});
	inputVertexParams.enabledBindingFlags = core::createBitmask({ EE_POSITIONS, EE_VELOCITIES, EE_COLORS, EE_COLORS_RISING_FLAG });

	for (uint8_t i = 0; i < EE_COUNT; ++i)
	{
		inputVertexParams.bindings[i].stride = (i == EE_COLORS_RISING_FLAG ? getTexelOrBlockBytesize(EF_R8G8B8A8_UINT) : getTexelOrBlockBytesize(EF_R32G32B32A32_SFLOAT));
		inputVertexParams.bindings[i].inputRate = asset::EVIR_PER_VERTEX;

		inputVertexParams.attributes[i].binding = i;
		inputVertexParams.attributes[i].format = (i == EE_COLORS_RISING_FLAG ? EF_R8G8B8A8_UINT : asset::EF_R32G32B32A32_SFLOAT);
		inputVertexParams.attributes[i].relativeOffset = 0;
	}

	asset::SBlendParams blendParams;
	asset::SPrimitiveAssemblyParams primitiveAssemblyParams;
	primitiveAssemblyParams.primitiveType = EPT_POINT_LIST;
	asset::SRasterizationParams rasterizationParams;

	video::IGPUDescriptorSetLayout::SBinding gpuUboBinding;
	gpuUboBinding.count = 1u;
	gpuUboBinding.binding = 0;
	gpuUboBinding.stageFlags = static_cast<asset::ICPUSpecializedShader::E_SHADER_STAGE>(asset::ICPUSpecializedShader::ESS_VERTEX | asset::ICPUSpecializedShader::ESS_FRAGMENT);
	gpuUboBinding.type = asset::EDT_UNIFORM_BUFFER;

	auto gpuGDescriptorPool = createDescriptorPool(1, EDT_UNIFORM_BUFFER);
	auto gpuGDs1Layout = logicalDevice->createGPUDescriptorSetLayout(&gpuUboBinding, &gpuUboBinding + 1);
	auto gpuUBO = logicalDevice->createDeviceLocalGPUBufferOnDedMem(sizeof(SBasicViewParameters));

	auto gpuGDescriptorSet1 = logicalDevice->createGPUDescriptorSet(gpuGDescriptorPool.get(), gpuGDs1Layout);
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
		logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);
	}

	auto vertexShaderBundle = assetManager->getAsset("../vertexShader.vert", {});
	{
		bool status = !vertexShaderBundle.getContents().empty();
		assert(status);
	}
	
	auto cpuVertexShader = core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(vertexShaderBundle.getContents().begin()[0]);
	smart_refctd_ptr<video::IGPUSpecializedShader> gpuVertexShader;
	{
		auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&cpuVertexShader, &cpuVertexShader + 1, cpu2gpuParams);
		if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
			assert(false);

		gpuVertexShader = (*gpu_array)[0];
	}

	auto fragmentShaderBundle = assetManager->getAsset("../fragmentShader.frag", {});
	{
		bool status = !fragmentShaderBundle.getContents().empty();
		assert(status);
	}

	auto cpuFragmentShader = core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(fragmentShaderBundle.getContents().begin()[0]);
	smart_refctd_ptr<video::IGPUSpecializedShader> gpuFragmentShader;
	{
		auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&cpuFragmentShader, &cpuFragmentShader + 1, cpu2gpuParams);
		if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
			assert(false);

		gpuFragmentShader = (*gpu_array)[0];
	}

	auto geometryShaderBundle = assetManager->getAsset("../geometryShader.geom", {});
	{
		bool status = !geometryShaderBundle.getContents().empty();
		assert(status);
	}

	auto cpuGeometryShader = core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(geometryShaderBundle.getContents().begin()[0]);
	smart_refctd_ptr<video::IGPUSpecializedShader> gpuGeometryShader;
	{
		auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&cpuGeometryShader, &cpuGeometryShader + 1, cpu2gpuParams);
		if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
			assert(false);

		gpuGeometryShader = (*gpu_array)[0];
	}

	core::smart_refctd_ptr<video::IGPUSpecializedShader> gpuGShaders[] = { gpuVertexShader, gpuFragmentShader, gpuGeometryShader };
	auto gpuGShadersPointer = reinterpret_cast<video::IGPUSpecializedShader**>(gpuGShaders);

	auto gpuGPipelineLayout = logicalDevice->createGPUPipelineLayout(&pushConstantRange, &pushConstantRange + 1, nullptr, std::move(gpuGDs1Layout), nullptr, nullptr);
	auto gpuRenderpassIndependentPipeline = logicalDevice->createGPURenderpassIndependentPipeline(nullptr, core::smart_refctd_ptr(gpuGPipelineLayout), gpuGShadersPointer, gpuGShadersPointer + 2 /* discard geometry shader*/, inputVertexParams, blendParams, primitiveAssemblyParams, rasterizationParams);
	auto gpuRenderpassIndependentPipeline2 = logicalDevice->createGPURenderpassIndependentPipeline(nullptr, core::smart_refctd_ptr(gpuGPipelineLayout), gpuGShadersPointer, gpuGShadersPointer + 3, inputVertexParams, blendParams, primitiveAssemblyParams, rasterizationParams);

	asset::SBufferBinding<video::IGPUBuffer> gpuGbindings[video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];

	gpuGbindings[EE_POSITIONS].buffer = gpuSSBOBuffer;
	gpuGbindings[EE_POSITIONS].offset = 0;

	gpuGbindings[EE_VELOCITIES].buffer = gpuSSBOBuffer;
	gpuGbindings[EE_VELOCITIES].offset = sizeof(SShaderStorageBufferObject::positions);

	gpuGbindings[EE_COLORS].buffer = gpuSSBOBuffer;
	gpuGbindings[EE_COLORS].offset = gpuGbindings[EE_VELOCITIES].offset + sizeof(SShaderStorageBufferObject::velocities);

	gpuGbindings[EE_COLORS_RISING_FLAG].buffer = gpuSSBOBuffer;
	gpuGbindings[EE_COLORS_RISING_FLAG].offset = gpuGbindings[EE_COLORS].offset + sizeof(SShaderStorageBufferObject::colors);

	auto gpuMeshBuffer = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(std::move(gpuRenderpassIndependentPipeline), nullptr, gpuGbindings, asset::SBufferBinding<video::IGPUBuffer>());
	{
		gpuMeshBuffer->setIndexType(asset::EIT_UNKNOWN);
		gpuMeshBuffer->setIndexCount(NUMBER_OF_PARTICLES);
	}

	smart_refctd_ptr<video::IGPUGraphicsPipeline> gpuGraphicsPipeline;
	{
		nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
		graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(const_cast<video::IGPURenderpassIndependentPipeline*>(gpuMeshBuffer->getPipeline()));
		graphicsPipelineParams.renderpass = core::smart_refctd_ptr(renderpass);
		gpuGraphicsPipeline = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));
	}

	auto gpuMeshBuffer2 = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(std::move(gpuRenderpassIndependentPipeline2), nullptr, gpuGbindings, asset::SBufferBinding<video::IGPUBuffer>());
	{
		gpuMeshBuffer2->setIndexType(asset::EIT_UNKNOWN);
		gpuMeshBuffer2->setIndexCount(NUMBER_OF_PARTICLES);
	}

	smart_refctd_ptr<video::IGPUGraphicsPipeline> gpuGraphicsPipeline2;
	{
		nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
		graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(const_cast<video::IGPURenderpassIndependentPipeline*>(gpuMeshBuffer2->getPipeline()));
		graphicsPipelineParams.renderpass = core::smart_refctd_ptr(renderpass);
		gpuGraphicsPipeline2 = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));
	}

	SPushConstants pushConstants;

	const std::string captionData = "[Nabla Engine] Compute Shaders";
	window->setCaption(captionData);

	CEventReceiver eventReceiver;
	CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
	CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

	core::vectorSIMDf cameraPosition(0, 0, 0);
	matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60), float(WIN_W) / WIN_H, 0.001, 1000);
	Camera camera = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), projectionMatrix, 10.f, 1.f);
	auto lastTime = std::chrono::system_clock::now();

	constexpr size_t NBL_FRAMES_TO_AVERAGE = 100ull;
	size_t frame_count = 0ull;
	double time_sum = 0;
	double dtList[NBL_FRAMES_TO_AVERAGE] = {};
	for (size_t i = 0ull; i < NBL_FRAMES_TO_AVERAGE; ++i)
		dtList[i] = 0.0;

	nbl::core::smart_refctd_ptr<nbl::video::IGPUSemaphore> render_finished_sem;
	while (windowCallback->isWindowOpen())
	{
		auto renderStart = std::chrono::system_clock::now();
		const auto renderDt = std::chrono::duration_cast<std::chrono::milliseconds>(renderStart - lastTime).count();
		lastTime = renderStart;
		{ // Calculate Simple Moving Average for FrameTime
			time_sum -= dtList[frame_count];
			time_sum += renderDt;
			dtList[frame_count] = renderDt;
			frame_count++;
			if (frame_count >= NBL_FRAMES_TO_AVERAGE)
				frame_count = 0;
		}
		const double averageFrameTime = time_sum / (double)NBL_FRAMES_TO_AVERAGE;

		#ifdef NBL_MORE_LOGS
				logger->log("renderDt = %f ------ averageFrameTime = %f", system::ILogger::ELL_INFO, renderDt, averageFrameTime);
		#endif // NBL_MORE_LOGS

		auto averageFrameTimeDuration = std::chrono::duration<double, std::milli>(averageFrameTime);
		auto nextPresentationTime = renderStart + averageFrameTimeDuration;
		auto nextPresentationTimeStamp = std::chrono::duration_cast<std::chrono::microseconds>(nextPresentationTime.time_since_epoch());

		inputSystem->getDefaultMouse(&mouse);
		inputSystem->getDefaultKeyboard(&keyboard);

		camera.beginInputProcessing(nextPresentationTimeStamp);
		mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); }, logger.get());
		keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); eventReceiver.process(events); }, logger.get());
		camera.endInputProcessing(nextPresentationTimeStamp);

		const auto& viewMatrix = camera.getViewMatrix();
		const auto& viewProjectionMatrix = camera.getConcatenatedMatrix();

		commandBuffer->reset(nbl::video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
		commandBuffer->begin(0);

		asset::SViewport viewport;
		viewport.minDepth = 1.f;
		viewport.maxDepth = 0.f;
		viewport.x = 0u;
		viewport.y = 0u;
		viewport.width = WIN_W;
		viewport.height = WIN_H;
		commandBuffer->setViewport(0u, 1u, &viewport);

		nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
		nbl::asset::VkRect2D area;
		area.offset = { 0,0 };
		area.extent = { WIN_W, WIN_H };
		nbl::asset::SClearValue clear[2];
		clear[0].color.float32[0] = 0.f;
		clear[0].color.float32[1] = 0.f;
		clear[0].color.float32[2] = 0.f;
		clear[0].color.float32[3] = 0.f;
		clear[1].depthStencil.depth = 0.f;

		beginInfo.clearValueCount = 1u;
		beginInfo.framebuffer = fbo;
		beginInfo.renderpass = renderpass;
		beginInfo.renderArea = area;
		beginInfo.clearValues = clear;

		commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);

		pushConstants.isXPressed = eventReceiver.isXPressed();
		pushConstants.isZPressed = eventReceiver.isZPressed();
		pushConstants.isCPressed = eventReceiver.isCPressed();	
		pushConstants.currentUserAbsolutePosition = camera.getPosition().getAsVector3df();

		/*
			Calculation of particle postitions takes place here
		*/

		commandBuffer->bindComputePipeline(gpuComputePipeline.get());
		commandBuffer->pushConstants(gpuComputePipeline->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0, sizeof(SPushConstants), &pushConstants);
		commandBuffer->bindDescriptorSets(EPBP_COMPUTE, gpuComputePipeline->getLayout(), 0, 1, &gpuCDescriptorSet.get(), nullptr);

		static_assert(NUMBER_OF_PARTICLES % WORK_GROUP_SIZE == 0, "Inccorect amount!");
		_NBL_STATIC_INLINE_CONSTEXPR size_t groupCountX = NUMBER_OF_PARTICLES / WORK_GROUP_SIZE;

		commandBuffer->dispatch(groupCountX, 1, 1);
	
		/*
			After calculation of positions each particle gets displayed
		*/

		core::matrix3x4SIMD modelMatrix;
		modelMatrix.setTranslation(nbl::core::vectorSIMDf(0, 0, 0, 0));

		core::matrix4SIMD mvp = core::concatenateBFollowedByA(viewProjectionMatrix, modelMatrix);

		SBasicViewParameters uboData;
		memcpy(uboData.MV, viewMatrix.pointer(), sizeof(uboData.MV));
		memcpy(uboData.MVP, mvp.pointer(), sizeof(uboData.MVP));
		memcpy(uboData.NormalMat, viewMatrix.pointer(), sizeof(uboData.NormalMat));
		commandBuffer->updateBuffer(gpuUBO.get(), 0ull, sizeof(uboData), &uboData);

		/*
			Draw particles
		*/

		commandBuffer->bindGraphicsPipeline(gpuGraphicsPipeline.get());
		commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuMeshBuffer->getPipeline()->getLayout(), 1u, 1u, &gpuGDescriptorSet1.get(), nullptr);
		commandBuffer->drawMeshBuffer(gpuMeshBuffer.get());

		/*
			Draw extras with geometry usage under key c and v conditions
		*/

		commandBuffer->bindGraphicsPipeline(gpuGraphicsPipeline2.get());
		commandBuffer->pushConstants(gpuMeshBuffer2->getPipeline()->getLayout(), asset::ISpecializedShader::ESS_GEOMETRY, 0, sizeof(SPushConstants), &pushConstants);
		commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuMeshBuffer2->getPipeline()->getLayout(), 1u, 1u, &gpuGDescriptorSet1.get(), nullptr);
		commandBuffer->drawMeshBuffer(gpuMeshBuffer2.get());

		commandBuffer->endRenderPass();
		commandBuffer->end();

		auto img_acq_sem = logicalDevice->createSemaphore();
		render_finished_sem = logicalDevice->createSemaphore();

		uint32_t imgnum = 0u;
		constexpr uint64_t MAX_TIMEOUT = 99999999999999ull; // ns
		swapchain->acquireNextImage(MAX_TIMEOUT, img_acq_sem.get(), nullptr, &imgnum);

		CommonAPI::Submit(logicalDevice.get(), swapchain.get(), commandBuffer.get(), queues[decltype(initOutput)::EQT_GRAPHICS], img_acq_sem.get(), render_finished_sem.get());
		CommonAPI::Present(logicalDevice.get(), swapchain.get(), queues[decltype(initOutput)::EQT_GRAPHICS], render_finished_sem.get(), imgnum);
	}

	const auto& fboCreationParams = fbo->getCreationParameters();
	auto gpuSourceImageView = fboCreationParams.attachments[0];

	bool status = ext::ScreenShot::createScreenShot(logicalDevice.get(), queues[decltype(initOutput)::EQT_TRANSFER_UP], render_finished_sem.get(), gpuSourceImageView.get(), assetManager.get(), "ScreenShot.png");
	assert(status);

	return 0;
}
