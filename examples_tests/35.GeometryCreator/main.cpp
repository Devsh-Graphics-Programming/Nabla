// Copyright (C) 2018-2021 - DevSH Graphics Programming Sp. z O.O.
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
using namespace core;

/*
	Uncomment for more detailed logging
*/

// #define NBL_MORE_LOGS

const char* vertexSource = R"===(
#version 430 core
layout(location = 0) in vec4 vPos;
layout(location = 3) in vec3 vNormal;

#include <nbl/builtin/glsl/utils/common.glsl>
#include <nbl/builtin/glsl/utils/transform.glsl>

layout( push_constant, row_major ) uniform Block {
	mat4 modelViewProj;
} PushConstants;

layout(location = 0) out vec3 Color; //per vertex output color, will be interpolated across the triangle

void main()
{
    gl_Position = PushConstants.modelViewProj*vPos;
    Color = vNormal*0.5+vec3(0.5);
}
)===";

const char* fragmentSource = R"===(
#version 430 core

layout(location = 0) in vec3 Color;

layout(location = 0) out vec4 pixelColor;

void main()
{
    pixelColor = vec4(Color,1.0);
}
)===";

#include "nbl/nblpack.h"
struct GPUObject
{
	core::smart_refctd_ptr<video::IGPUMeshBuffer> gpuMeshbBuffer;
	core::smart_refctd_ptr<video::IGPUGraphicsPipeline> gpuGraphicsPipeline;
} PACK_STRUCT;

struct Objects
{
	enum E_OBJECT_INDEX
	{
		E_CUBE,
		E_SPHERE,
		E_CYLINDER,
		E_RECTANGLE,
		E_DISK,
		E_CONE,
		E_ARROW,
		E_ICOSPHERE,
		E_COUNT
	};

	Objects() = default;
	Objects(std::initializer_list<std::pair<asset::IGeometryCreator::return_type, GPUObject>> _objects) : objects(_objects) {}

	std::vector<std::pair<asset::IGeometryCreator::return_type, GPUObject>> objects;
} PACK_STRUCT;
#include "nbl/nblunpack.h"

class GeometryCreatorApp : public ApplicationBase
{
	static constexpr uint32_t WIN_W = 1280;
	static constexpr uint32_t WIN_H = 720;
	static constexpr uint32_t SC_IMG_COUNT = 3u;
	static constexpr uint32_t FRAMES_IN_FLIGHT = 5u;
	static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
	static constexpr size_t NBL_FRAMES_TO_AVERAGE = 100ull;
	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

public:
	struct Nabla : IUserData
	{
		nbl::core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
		nbl::core::smart_refctd_ptr<nbl::ui::IWindow> window;
		nbl::core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCallback;
		nbl::core::smart_refctd_ptr<nbl::video::IAPIConnection> gl;
		nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface;
		nbl::core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
		nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
		nbl::video::IPhysicalDevice* gpuPhysicalDevice;
		std::array<nbl::video::IGPUQueue*, CommonAPI::InitOutput<SC_IMG_COUNT>::EQT_COUNT> queues = { nullptr, nullptr, nullptr, nullptr };
		nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
		std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, SC_IMG_COUNT> fbos;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool> commandPool;
		nbl::core::smart_refctd_ptr<nbl::system::ISystem> system;
		nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
		nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
		nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger;
		nbl::core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;

		nbl::core::smart_refctd_ptr<video::IGPUFence> gpuTransferFence;
		nbl::core::smart_refctd_ptr<video::IGPUFence> gpuComputeFence;
		nbl::video::IGPUObjectFromAssetConverter cpu2gpu;

		CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
		CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

		core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
		core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
		core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
		uint64_t asdf[8u];
		core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FRAMES_IN_FLIGHT];

		Camera camera = Camera(core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 0, 0), core::matrix4SIMD());
		std::chrono::system_clock::time_point lastTime;

		bool frameDataFilled = false;
		size_t frame_count = 0ull;
		double time_sum = 0;
		double dtList[NBL_FRAMES_TO_AVERAGE] = {};

		uint32_t acquiredNextFBO = {};
		int resourceIx = -1;

		Objects cpuGpuObjects;

		void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override
		{
			window = std::move(wnd);
		}
	};

	APP_CONSTRUCTOR(GeometryCreatorApp)

	void onAppInitialized_impl(void* data) override
	{
		Nabla* engine = static_cast<Nabla*>(data);

		CommonAPI::InitOutput<SC_IMG_COUNT> initOutput;
		auto window = core::smart_refctd_ptr(initOutput.window);
		CommonAPI::Init<WIN_W, WIN_H, SC_IMG_COUNT>(initOutput, video::EAT_OPENGL, "MeshLoaders", nbl::asset::EF_D32_SFLOAT);
		engine->gl = std::move(initOutput.apiConnection);
		engine->surface = std::move(initOutput.surface);
		engine->gpuPhysicalDevice = std::move(initOutput.physicalDevice);
		engine->logicalDevice = std::move(initOutput.logicalDevice);
		engine->queues = std::move(initOutput.queues);
		engine->swapchain = std::move(initOutput.swapchain);
		engine->renderpass = std::move(initOutput.renderpass);
		engine->fbos = std::move(initOutput.fbo);
		engine->commandPool = std::move(initOutput.commandPool);
		engine->assetManager = std::move(initOutput.assetManager);
		engine->logger = std::move(initOutput.logger);
		engine->inputSystem = std::move(initOutput.inputSystem);
		engine->windowCallback = std::move(initOutput.windowCb);
		engine->cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
		engine->utilities = std::move(initOutput.utilities);

		engine->gpuTransferFence = engine->logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
		engine->gpuComputeFence = engine->logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

		nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
		{
			engine->cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].fence = &engine->gpuTransferFence;
			engine->cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].fence = &engine->gpuComputeFence;
		}

		auto cpu2gpuWaitForFences = [&]() -> void
		{
			video::IGPUFence::E_STATUS waitStatus = video::IGPUFence::ES_NOT_READY;
			while (waitStatus != video::IGPUFence::ES_SUCCESS)
			{
				waitStatus = engine->logicalDevice->waitForFences(1u, &engine->gpuTransferFence.get(), false, 999999999ull);
				if (waitStatus == video::IGPUFence::ES_ERROR)
					assert(false);
				else if (waitStatus == video::IGPUFence::ES_TIMEOUT)
					break;
			}

			waitStatus = video::IGPUFence::ES_NOT_READY;
			while (waitStatus != video::IGPUFence::ES_SUCCESS)
			{
				waitStatus = engine->logicalDevice->waitForFences(1u, &engine->gpuComputeFence.get(), false, 999999999ull);
				if (waitStatus == video::IGPUFence::ES_ERROR)
					assert(false);
				else if (waitStatus == video::IGPUFence::ES_TIMEOUT)
					break;
			}
		};

		auto geometryCreator = engine->assetManager->getGeometryCreator();
		auto cubeGeometry = geometryCreator->createCubeMesh(vector3df(2, 2, 2));
		auto sphereGeometry = geometryCreator->createSphereMesh(2, 16, 16);
		auto cylinderGeometry = geometryCreator->createCylinderMesh(2, 2, 20);
		auto rectangleGeometry = geometryCreator->createRectangleMesh(nbl::core::vector2df_SIMD(1.5, 3));
		auto diskGeometry = geometryCreator->createDiskMesh(2, 30);
		auto coneGeometry = geometryCreator->createConeMesh(2, 3, 10);
		auto arrowGeometry = geometryCreator->createArrowMesh();
		auto icosphereGeometry = geometryCreator->createIcoSphere(1, 3, true);

		auto createGPUSpecializedShaderFromSource = [=](const char* source, asset::ISpecializedShader::E_SHADER_STAGE stage) -> core::smart_refctd_ptr<video::IGPUSpecializedShader>
		{
			auto spirv = engine->assetManager->getGLSLCompiler()->createSPIRVFromGLSL(source, stage, "main", "runtimeID");
			if (!spirv)
				return nullptr;

			auto gpuUnspecializedShader = engine->logicalDevice->createGPUShader(std::move(spirv));
			return engine->logicalDevice->createGPUSpecializedShader(gpuUnspecializedShader.get(), { nullptr, nullptr, "main", stage });
		};

		auto createGPUSpecializedShaderFromSourceWithIncludes = [&](const char* source, asset::ISpecializedShader::E_SHADER_STAGE stage, const char* origFilepath)
		{
			auto resolved_includes = engine->assetManager->getGLSLCompiler()->resolveIncludeDirectives(source, stage, origFilepath);
			return createGPUSpecializedShaderFromSource(reinterpret_cast<const char*>(resolved_includes->getSPVorGLSL()->getPointer()), stage);
		};

		core::smart_refctd_ptr<video::IGPUSpecializedShader> gpuShaders[2] =
		{
			createGPUSpecializedShaderFromSourceWithIncludes(vertexSource,asset::ISpecializedShader::ESS_VERTEX, "shader.vert"),
			createGPUSpecializedShaderFromSource(fragmentSource,asset::ISpecializedShader::ESS_FRAGMENT)
		};
		auto gpuShadersRaw = reinterpret_cast<video::IGPUSpecializedShader**>(gpuShaders);

		auto createGPUMeshBufferAndItsPipeline = [&](asset::IGeometryCreator::return_type& geometryObject) -> GPUObject
		{
			asset::SBlendParams blendParams;
			asset::SRasterizationParams rasterParams;
			rasterParams.faceCullingMode = asset::EFCM_NONE;

			asset::SPushConstantRange range[1] = { asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD) };
			auto gpuRenderpassIndependentPipeline = engine->logicalDevice->createGPURenderpassIndependentPipeline
			(
				nullptr,
				engine->logicalDevice->createGPUPipelineLayout(range, range + 1u, nullptr, nullptr, nullptr, nullptr),
				gpuShadersRaw,
				gpuShadersRaw + sizeof(gpuShaders) / sizeof(core::smart_refctd_ptr<video::IGPUSpecializedShader>),
				geometryObject.inputParams,
				blendParams,
				geometryObject.assemblyParams,
				rasterParams
			);

			constexpr auto MAX_ATTR_BUF_BINDING_COUNT = video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT;
			constexpr auto MAX_DATA_BUFFERS = MAX_ATTR_BUF_BINDING_COUNT + 1;
			core::vector<asset::ICPUBuffer*> cpubuffers;
			cpubuffers.reserve(MAX_DATA_BUFFERS);
			for (auto i = 0; i < MAX_ATTR_BUF_BINDING_COUNT; i++)
			{
				auto buf = geometryObject.bindings[i].buffer.get();
				if (buf)
					cpubuffers.push_back(buf);
			}
			auto cpuindexbuffer = geometryObject.indexBuffer.buffer.get();
			if (cpuindexbuffer)
				cpubuffers.push_back(cpuindexbuffer);

			auto gpubuffers = cpu2gpu.getGPUObjectsFromAssets(cpubuffers.data(), cpubuffers.data() + cpubuffers.size(), engine->cpu2gpuParams);
			{
				if (!gpubuffers || gpubuffers->size() < 1u)
					assert(false);

				cpu2gpuWaitForFences();
			}

			asset::SBufferBinding<video::IGPUBuffer> bindings[MAX_DATA_BUFFERS];
			for (auto i = 0, j = 0; i < MAX_ATTR_BUF_BINDING_COUNT; i++)
			{
				if (!geometryObject.bindings[i].buffer)
					continue;
				auto buffPair = gpubuffers->operator[](j++);
				bindings[i].offset = buffPair->getOffset();
				bindings[i].buffer = core::smart_refctd_ptr<video::IGPUBuffer>(buffPair->getBuffer());
			}
			if (cpuindexbuffer)
			{
				auto buffPair = gpubuffers->back();
				bindings[MAX_ATTR_BUF_BINDING_COUNT].offset = buffPair->getOffset();
				bindings[MAX_ATTR_BUF_BINDING_COUNT].buffer = core::smart_refctd_ptr<video::IGPUBuffer>(buffPair->getBuffer());
			}

			auto mb = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(core::smart_refctd_ptr(gpuRenderpassIndependentPipeline), nullptr, bindings, std::move(bindings[MAX_ATTR_BUF_BINDING_COUNT]));
			{
				mb->setIndexType(geometryObject.indexType);
				mb->setIndexCount(geometryObject.indexCount);
				mb->setBoundingBox(geometryObject.bbox);
			}

			nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
			graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr(gpuRenderpassIndependentPipeline);
			graphicsPipelineParams.renderpass = core::smart_refctd_ptr(engine->renderpass);

			auto gpuGraphicsPipeline = engine->logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));

			return { mb, gpuGraphicsPipeline };
		};

		auto gpuCube = createGPUMeshBufferAndItsPipeline(cubeGeometry);
		auto gpuSphere = createGPUMeshBufferAndItsPipeline(sphereGeometry);
		auto gpuCylinder = createGPUMeshBufferAndItsPipeline(cylinderGeometry);
		auto gpuRectangle = createGPUMeshBufferAndItsPipeline(rectangleGeometry);
		auto gpuDisk = createGPUMeshBufferAndItsPipeline(diskGeometry);
		auto gpuCone = createGPUMeshBufferAndItsPipeline(coneGeometry);
		auto gpuArrow = createGPUMeshBufferAndItsPipeline(arrowGeometry);
		auto gpuIcosphere = createGPUMeshBufferAndItsPipeline(icosphereGeometry);

		Objects objects =
		{
			std::make_pair(cubeGeometry, gpuCube),
			std::make_pair(sphereGeometry, gpuSphere),
			std::make_pair(cylinderGeometry, gpuCylinder),
			std::make_pair(rectangleGeometry, gpuRectangle),
			std::make_pair(diskGeometry, gpuDisk),
			std::make_pair(coneGeometry, gpuCone),
			std::make_pair(arrowGeometry, gpuArrow),
			std::make_pair(icosphereGeometry, gpuIcosphere)
		};

		engine->cpuGpuObjects = std::move(objects);

		core::vectorSIMDf cameraPosition(0, 5, -10);
		matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60), float(WIN_W) / WIN_H, 0.001, 1000);
		engine->camera = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), projectionMatrix, 10.f, 1.f);
		engine->lastTime = std::chrono::system_clock::now();

		for (size_t i = 0ull; i < NBL_FRAMES_TO_AVERAGE; ++i)
			engine->dtList[i] = 0.0;

		engine->logicalDevice->createCommandBuffers(engine->commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, engine->commandBuffers);

		engine->frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
		engine->imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
		engine->renderFinished[FRAMES_IN_FLIGHT] = { nullptr };

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
		{
			engine->imageAcquire[i] = engine->logicalDevice->createSemaphore();
			engine->renderFinished[i] = engine->logicalDevice->createSemaphore();
		}
	}

	void onAppTerminated_impl(void* data) override
	{
		Nabla* engine = static_cast<Nabla*>(data);

		const auto& fboCreationParams = engine->fbos[engine->acquiredNextFBO]->getCreationParameters();
		auto gpuSourceImageView = fboCreationParams.attachments[0];

		bool status = ext::ScreenShot::createScreenShot(engine->logicalDevice.get(), engine->queues[CommonAPI::InitOutput<1>::EQT_TRANSFER_UP], engine->renderFinished[engine->resourceIx].get(), gpuSourceImageView.get(), engine->assetManager.get(), "ScreenShot.png");
		assert(status);
	}

	void workLoopBody(void* data) override
	{
		Nabla* engine = static_cast<Nabla*>(data);

		++engine->resourceIx;
		if (engine->resourceIx >= FRAMES_IN_FLIGHT)
			engine->resourceIx = 0;

		auto& commandBuffer = engine->commandBuffers[engine->resourceIx];
		auto& fence = engine->frameComplete[engine->resourceIx];

		if (fence)
			while (engine->logicalDevice->waitForFences(1u, &fence.get(), false, MAX_TIMEOUT) == video::IGPUFence::ES_TIMEOUT) {}
		else
			fence = engine->logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

		auto renderStart = std::chrono::system_clock::now();
		const auto renderDt = std::chrono::duration_cast<std::chrono::milliseconds>(renderStart - engine->lastTime).count();
		engine->lastTime = renderStart;
		{ // Calculate Simple Moving Average for FrameTime
			engine->time_sum -= engine->dtList[engine->frame_count];
			engine->time_sum += renderDt;
			engine->dtList[engine->frame_count] = renderDt;
			engine->frame_count++;
			if (engine->frame_count >= NBL_FRAMES_TO_AVERAGE)
			{
				engine->frameDataFilled = true;
				engine->frame_count = 0;
			}

		}
		const double averageFrameTime = engine->frameDataFilled ? (engine->time_sum / (double)NBL_FRAMES_TO_AVERAGE) : (engine->time_sum / engine->frame_count);

#ifdef NBL_MORE_LOGS
		logger->log("renderDt = %f ------ averageFrameTime = %f", system::ILogger::ELL_INFO, renderDt, averageFrameTime);
#endif // NBL_MORE_LOGS

		auto averageFrameTimeDuration = std::chrono::duration<double, std::milli>(averageFrameTime);
		auto nextPresentationTime = renderStart + averageFrameTimeDuration;
		auto nextPresentationTimeStamp = std::chrono::duration_cast<std::chrono::microseconds>(nextPresentationTime.time_since_epoch());

		engine->inputSystem->getDefaultMouse(&engine->mouse);
		engine->inputSystem->getDefaultKeyboard(&engine->keyboard);

		engine->camera.beginInputProcessing(nextPresentationTimeStamp);
		engine->mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { engine->camera.mouseProcess(events); }, engine->logger.get());
		engine->keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { engine->camera.keyboardProcess(events); }, engine->logger.get());
		engine->camera.endInputProcessing(nextPresentationTimeStamp);

		const auto& viewMatrix = engine->camera.getViewMatrix();
		const auto& viewProjectionMatrix = engine->camera.getConcatenatedMatrix();

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

		engine->swapchain->acquireNextImage(MAX_TIMEOUT, engine->imageAcquire[engine->resourceIx].get(), nullptr, &engine->acquiredNextFBO);

		nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
		{
			VkRect2D area;
			area.offset = { 0,0 };
			area.extent = { WIN_W, WIN_H };
			asset::SClearValue clear[2] = {};
			clear[0].color.float32[0] = 1.f;
			clear[0].color.float32[1] = 1.f;
			clear[0].color.float32[2] = 1.f;
			clear[0].color.float32[3] = 1.f;
			clear[1].depthStencil.depth = 0.f;

			beginInfo.clearValueCount = 2u;
			beginInfo.framebuffer = engine->fbos[engine->acquiredNextFBO];
			beginInfo.renderpass = engine->renderpass;
			beginInfo.renderArea = area;
			beginInfo.clearValues = clear;
		}

		commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);

		for (auto index = 0u; index < engine->cpuGpuObjects.objects.size(); ++index)
		{
			const auto iterator = engine->cpuGpuObjects.objects[index];
			auto geometryObject = iterator.first;
			auto gpuObject = iterator.second;

			core::matrix3x4SIMD modelMatrix;
			modelMatrix.setTranslation(nbl::core::vectorSIMDf(index * 5, 0, 0, 0));

			core::matrix4SIMD mvp = core::concatenateBFollowedByA(viewProjectionMatrix, modelMatrix);
			auto* gpuGraphicsPipeline = gpuObject.gpuGraphicsPipeline.get();

			commandBuffer->bindGraphicsPipeline(gpuGraphicsPipeline);
			commandBuffer->pushConstants(gpuGraphicsPipeline->getRenderpassIndependentPipeline()->getLayout(), video::IGPUSpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), mvp.pointer());
			commandBuffer->drawMeshBuffer(gpuObject.gpuMeshbBuffer.get());
		}

		commandBuffer->endRenderPass();
		commandBuffer->end();

		CommonAPI::Submit(engine->logicalDevice.get(), engine->swapchain.get(), commandBuffer.get(), engine->queues[CommonAPI::InitOutput<1>::EQT_GRAPHICS], engine->imageAcquire[engine->resourceIx].get(), engine->renderFinished[engine->resourceIx].get(), fence.get());
		CommonAPI::Present(engine->logicalDevice.get(), engine->swapchain.get(), engine->queues[CommonAPI::InitOutput<1>::EQT_GRAPHICS], engine->renderFinished[engine->resourceIx].get(), engine->acquiredNextFBO);
	}

	bool keepRunning(void* params) override
	{
		Nabla* engine = static_cast<Nabla*>(params);
		return engine->windowCallback->isWindowOpen();
	}
};

NBL_COMMON_API_MAIN(GeometryCreatorApp, GeometryCreatorApp::Nabla)