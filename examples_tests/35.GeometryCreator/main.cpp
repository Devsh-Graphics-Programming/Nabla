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

#define NBL_MORE_LOGS

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

	Objects(std::initializer_list<std::pair<asset::IGeometryCreator::return_type, GPUObject>> _objects) : objects(_objects) {}

	const std::vector<std::pair<asset::IGeometryCreator::return_type, GPUObject>> objects;
} PACK_STRUCT;
#include "nbl/nblunpack.h"

const char* vertexSource = R"===(
#version 430 core
layout(location = 0) in vec4 vPos;
layout(location = 1) in vec4 vColor;
layout(location = 2) in vec2 vUv;
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

const char* vertexSource_cone = R"===(
#version 430 core
layout(location = 0) in vec4 vPos;
layout(location = 1) in vec4 vColor;
layout(location = 2) in vec3 vNormal;

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

const char* vertexSource_ico = R"===(
#version 430 core
layout(location = 0) in vec4 vPos;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vUv;

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

class GeometryCreatorSampleApp : public ApplicationBase
{
	constexpr static uint32_t WIN_W = 1280;
	constexpr static uint32_t WIN_H = 720;
	constexpr static uint32_t SC_IMG_COUNT = 3u;
	constexpr static uint32_t FRAMES_IN_FLIGHT = 5u;
	static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
	static constexpr size_t NBL_FRAMES_TO_AVERAGE = 100ull;
	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

	core::smart_refctd_ptr<nbl::ui::IWindow> window;
	core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
	core::smart_refctd_ptr<nbl::system::ILogger> logger;
	core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;
	core::smart_refctd_ptr<nbl::video::IAPIConnection> api;
	core::smart_refctd_ptr<nbl::video::ISurface> surface;
	video::IPhysicalDevice* gpuPhysicalDevice;
	core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
	std::array<video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> queues;
	core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
	core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, CommonAPI::InitOutput::MaxSwapChainImageCount> fbos;
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxQueuesCount> commandPools;
	core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
	video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
	core::smart_refctd_ptr<nbl::video::IUtilities> utilities;

	int32_t m_resourceIx = -1;
	core::smart_refctd_ptr<video::IGPUFence> m_frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> m_imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> m_renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUCommandBuffer> m_commandBuffers[FRAMES_IN_FLIGHT];

	std::chrono::system_clock::time_point m_lastTime;
	double m_time_sum = 0;
	size_t m_frame_count = 0ull;
	double m_dtList[NBL_FRAMES_TO_AVERAGE] = {};
	bool m_frameDataFilled = false;
	uint32_t m_acquiredNextFBO = {};

	CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> m_mouse;
	CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> m_keyboard;
	std::unique_ptr<Camera> m_camera = nullptr;
	std::unique_ptr<Objects> m_cpuGpuObjects = nullptr;

public:
	void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override
	{
		window = std::move(wnd);
	}
	nbl::ui::IWindow* getWindow() override
	{
		return window.get();
	}
	void setSystem(core::smart_refctd_ptr<nbl::system::ISystem>&& system) override
	{
		system = std::move(system);
	}

	APP_CONSTRUCTOR(GeometryCreatorSampleApp);

	void onAppInitialized_impl() override
	{
		CommonAPI::SFeatureRequest<video::IAPIConnection::E_FEATURE> requiredInstanceFeatures = {};
		requiredInstanceFeatures.count = 1u;
		video::IAPIConnection::E_FEATURE requiredFeatures_Instance[] = { video::IAPIConnection::EF_SURFACE };
		requiredInstanceFeatures.features = requiredFeatures_Instance;

		CommonAPI::SFeatureRequest<video::IAPIConnection::E_FEATURE> optionalInstanceFeatures = {};

		CommonAPI::SFeatureRequest<video::ILogicalDevice::E_FEATURE> requiredDeviceFeatures = {};
		requiredDeviceFeatures.count = 1u;
		video::ILogicalDevice::E_FEATURE requiredFeatures_Device[] = { video::ILogicalDevice::EF_SWAPCHAIN };
		requiredDeviceFeatures.features = requiredFeatures_Device;

		CommonAPI::SFeatureRequest< video::ILogicalDevice::E_FEATURE> optionalDeviceFeatures = {};

		const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT);
		const video::ISurface::SFormat surfaceFormat(asset::EF_B8G8R8A8_SRGB, asset::ECP_COUNT, asset::EOTF_UNKNOWN);

		CommonAPI::InitOutput initOutput;
		initOutput.window = core::smart_refctd_ptr(window);
		CommonAPI::Init(
			initOutput,
			video::EAT_OPENGL,
			"35.GeometryCreator",
			requiredInstanceFeatures,
			optionalInstanceFeatures,
			requiredDeviceFeatures,
			optionalDeviceFeatures,
			WIN_W,
			WIN_H,
			SC_IMG_COUNT,
			swapchainImageUsage,
			surfaceFormat);

		window = std::move(initOutput.window);
		windowCb = std::move(initOutput.windowCb);
		logger = std::move(initOutput.logger);
		inputSystem = std::move(initOutput.inputSystem);
		api = std::move(initOutput.apiConnection);
		surface = std::move(initOutput.surface);
		gpuPhysicalDevice = std::move(initOutput.physicalDevice);
		logicalDevice = std::move(initOutput.logicalDevice);
		queues = std::move(initOutput.queues);
		swapchain = std::move(initOutput.swapchain);
		renderpass = std::move(initOutput.renderpass);
		fbos = std::move(initOutput.fbo);
		commandPools = std::move(initOutput.commandPools);
		assetManager = std::move(initOutput.assetManager);
		cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
		utilities = std::move(initOutput.utilities);

		auto geometryCreator = assetManager->getGeometryCreator();
		auto cubeGeometry = geometryCreator->createCubeMesh(vector3df(2, 2, 2));
		auto sphereGeometry = geometryCreator->createSphereMesh(2, 16, 16);
		auto cylinderGeometry = geometryCreator->createCylinderMesh(2, 2, 20);
		auto rectangleGeometry = geometryCreator->createRectangleMesh(nbl::core::vector2df_SIMD(1.5, 3));
		auto diskGeometry = geometryCreator->createDiskMesh(2, 30);
		auto coneGeometry = geometryCreator->createConeMesh(2, 3, 10);
		auto arrowGeometry = geometryCreator->createArrowMesh();
		auto icosphereGeometry = geometryCreator->createIcoSphere(1, 3, true);

		auto createGPUSpecializedShaderFromSource = [=](const char* source, asset::IShader::E_SHADER_STAGE stage) -> core::smart_refctd_ptr<video::IGPUSpecializedShader>
		{
			auto spirv = assetManager->getGLSLCompiler()->createSPIRVFromGLSL(source, stage, "main", "runtimeID", nullptr, true, nullptr, logger.get());
			if (!spirv)
				return nullptr;

			auto gpuUnspecializedShader = logicalDevice->createGPUShader(std::move(spirv));
			return logicalDevice->createGPUSpecializedShader(gpuUnspecializedShader.get(), { nullptr, nullptr, "main" });
		};

		auto createGPUSpecializedShaderFromSourceWithIncludes = [&](const char* source, asset::IShader::E_SHADER_STAGE stage, const char* origFilepath)
		{
			auto resolved_includes = assetManager->getGLSLCompiler()->resolveIncludeDirectives(source, stage, origFilepath);
			return createGPUSpecializedShaderFromSource(reinterpret_cast<const char*>(resolved_includes->getSPVorGLSL()->getPointer()), stage);
		};

		core::smart_refctd_ptr<video::IGPUSpecializedShader> gpuShaders[2] =
		{
			createGPUSpecializedShaderFromSourceWithIncludes(vertexSource,asset::IShader::ESS_VERTEX, "shader.vert"),
			createGPUSpecializedShaderFromSource(fragmentSource,asset::IShader::ESS_FRAGMENT)
		};
		auto gpuShadersRaw = reinterpret_cast<video::IGPUSpecializedShader**>(gpuShaders);

		core::smart_refctd_ptr<video::IGPUSpecializedShader> gpuShaders_cone[2] =
		{
			createGPUSpecializedShaderFromSourceWithIncludes(vertexSource_cone, asset::IShader::ESS_VERTEX, "shader_cone.vert"),
			gpuShaders[1]
		};
		auto gpuShadersRaw_cone = reinterpret_cast<video::IGPUSpecializedShader**>(gpuShaders_cone);

		core::smart_refctd_ptr<video::IGPUSpecializedShader> gpuShaders_ico[2] =
		{
			createGPUSpecializedShaderFromSourceWithIncludes(vertexSource_ico, asset::IShader::ESS_VERTEX, "shader_ico.vert"),
			gpuShaders[1]
		};
		auto gpuShadersRaw_ico = reinterpret_cast<video::IGPUSpecializedShader**>(gpuShaders_ico);

		auto createGPUMeshBufferAndItsPipeline = [&](asset::IGeometryCreator::return_type& geometryObject, Objects::E_OBJECT_INDEX object) -> GPUObject
		{
			asset::SBlendParams blendParams;
			blendParams.logicOpEnable = false;
			blendParams.logicOp = nbl::asset::ELO_NO_OP;
			for (size_t i = 0ull; i < nbl::asset::SBlendParams::MAX_COLOR_ATTACHMENT_COUNT; i++)
				blendParams.blendParams[i].attachmentEnabled = (i == 0ull);

			asset::SRasterizationParams rasterParams;
			rasterParams.faceCullingMode = asset::EFCM_NONE;

			asset::SPushConstantRange range[1] = { asset::IShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD) };
			core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> gpuRenderpassIndependentPipeline = nullptr;
			if (object == Objects::E_CONE)
			{
				gpuRenderpassIndependentPipeline = logicalDevice->createGPURenderpassIndependentPipeline
				(
					nullptr,
					logicalDevice->createGPUPipelineLayout(range, range + 1u, nullptr, nullptr, nullptr, nullptr),
					gpuShadersRaw_cone,
					gpuShadersRaw_cone + sizeof(gpuShaders_cone) / sizeof(core::smart_refctd_ptr<video::IGPUSpecializedShader>),
					geometryObject.inputParams,
					blendParams,
					geometryObject.assemblyParams,
					rasterParams
				);
			}
			else if (object == Objects::E_ICOSPHERE)
			{
				gpuRenderpassIndependentPipeline = logicalDevice->createGPURenderpassIndependentPipeline
				(
					nullptr,
					logicalDevice->createGPUPipelineLayout(range, range + 1u, nullptr, nullptr, nullptr, nullptr),
					gpuShadersRaw_ico,
					gpuShadersRaw_ico + sizeof(gpuShaders_ico) / sizeof(core::smart_refctd_ptr<video::IGPUSpecializedShader>),
					geometryObject.inputParams,
					blendParams,
					geometryObject.assemblyParams,
					rasterParams
				);
			}
			else
			{
				gpuRenderpassIndependentPipeline = logicalDevice->createGPURenderpassIndependentPipeline
				(
					nullptr,
					logicalDevice->createGPUPipelineLayout(range, range + 1u, nullptr, nullptr, nullptr, nullptr),
					gpuShadersRaw,
					gpuShadersRaw + sizeof(gpuShaders) / sizeof(core::smart_refctd_ptr<video::IGPUSpecializedShader>),
					geometryObject.inputParams,
					blendParams,
					geometryObject.assemblyParams,
					rasterParams
				);
			}

			constexpr auto MAX_ATTR_BUF_BINDING_COUNT = video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT;
			constexpr auto MAX_DATA_BUFFERS = MAX_ATTR_BUF_BINDING_COUNT + 1;
			core::vector<asset::ICPUBuffer*> cpubuffers;
			cpubuffers.reserve(MAX_DATA_BUFFERS);
			for (auto i = 0; i < MAX_ATTR_BUF_BINDING_COUNT; i++)
			{
				auto buf = geometryObject.bindings[i].buffer.get();
				if (buf)
				{
					const auto newUsageFlags = buf->getUsageFlags() | asset::IBuffer::EUF_VERTEX_BUFFER_BIT;
					buf->setUsageFlags(newUsageFlags);
					cpubuffers.push_back(buf);
				}
			}
			auto cpuindexbuffer = geometryObject.indexBuffer.buffer.get();
			if (cpuindexbuffer)
			{
				const auto newUsageFlags = cpuindexbuffer->getUsageFlags() | asset::IBuffer::EUF_INDEX_BUFFER_BIT;
				cpuindexbuffer->setUsageFlags(newUsageFlags);
				cpubuffers.push_back(cpuindexbuffer);
			}

			video::IGPUObjectFromAssetConverter cpu2gpu;

			core::smart_refctd_ptr<video::IGPUCommandBuffer> transferCmdBuffer;
			core::smart_refctd_ptr<video::IGPUCommandBuffer> computeCmdBuffer;

			logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_TRANSFER_UP].get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &transferCmdBuffer);
			logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_COMPUTE].get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &computeCmdBuffer);

			cpu2gpuParams.perQueue[video::IGPUObjectFromAssetConverter::EQU_TRANSFER].cmdbuf = transferCmdBuffer;
			cpu2gpuParams.perQueue[video::IGPUObjectFromAssetConverter::EQU_COMPUTE].cmdbuf = computeCmdBuffer;

			cpu2gpuParams.beginCommandBuffers();
			auto gpubuffers = cpu2gpu.getGPUObjectsFromAssets(cpubuffers.data(), cpubuffers.data() + cpubuffers.size(), cpu2gpuParams);
			cpu2gpuParams.waitForCreationToComplete(false);
			if (!gpubuffers || gpubuffers->size() < 1u)
				assert(false);

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
			graphicsPipelineParams.renderpass = core::smart_refctd_ptr(renderpass);

			auto gpuGraphicsPipeline = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));

			return { mb, gpuGraphicsPipeline };
		};

		auto gpuCube = createGPUMeshBufferAndItsPipeline(cubeGeometry, Objects::E_CUBE);
		auto gpuSphere = createGPUMeshBufferAndItsPipeline(sphereGeometry, Objects::E_SPHERE);
		auto gpuCylinder = createGPUMeshBufferAndItsPipeline(cylinderGeometry, Objects::E_CYLINDER);
		auto gpuRectangle = createGPUMeshBufferAndItsPipeline(rectangleGeometry, Objects::E_RECTANGLE);
		auto gpuDisk = createGPUMeshBufferAndItsPipeline(diskGeometry, Objects::E_DISK);
		auto gpuCone = createGPUMeshBufferAndItsPipeline(coneGeometry, Objects::E_CONE);
		auto gpuArrow = createGPUMeshBufferAndItsPipeline(arrowGeometry, Objects::E_ARROW);
		auto gpuIcosphere = createGPUMeshBufferAndItsPipeline(icosphereGeometry, Objects::E_ICOSPHERE);

		m_cpuGpuObjects = std::make_unique<Objects>
		(std::initializer_list({
			std::make_pair(cubeGeometry, gpuCube),
			std::make_pair(sphereGeometry, gpuSphere),
			std::make_pair(cylinderGeometry, gpuCylinder),
			std::make_pair(rectangleGeometry, gpuRectangle),
			std::make_pair(diskGeometry, gpuDisk),
			std::make_pair(coneGeometry, gpuCone),
			std::make_pair(arrowGeometry, gpuArrow),
			std::make_pair(icosphereGeometry, gpuIcosphere)
		}));

		core::vectorSIMDf cameraPosition(0, 5, -10);
		matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60), float(WIN_W) / WIN_H, 0.001, 1000);
		m_camera = std::make_unique<Camera>(cameraPosition, core::vectorSIMDf(0, 0, 0), projectionMatrix, 10.f, 1.f);
		m_lastTime = std::chrono::system_clock::now();
		
		for (size_t i = 0ull; i < NBL_FRAMES_TO_AVERAGE; ++i)
			m_dtList[i] = 0.0;

		logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, m_commandBuffers);

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
		{
			m_imageAcquire[i] = logicalDevice->createSemaphore();
			m_renderFinished[i] = logicalDevice->createSemaphore();
		}
	}

	void onAppTerminated_impl() override
	{
		logicalDevice->waitIdle();

		const auto& fboCreationParams = fbos[m_acquiredNextFBO]->getCreationParameters();
		auto gpuSourceImageView = fboCreationParams.attachments[0];

		bool status = ext::ScreenShot::createScreenShot(
			logicalDevice.get(),
			queues[CommonAPI::InitOutput::EQT_TRANSFER_UP],
			m_renderFinished[m_resourceIx].get(),
			gpuSourceImageView.get(),
			assetManager.get(),
			"ScreenShot.png",
			asset::EIL_PRESENT_SRC_KHR,
			static_cast<asset::E_ACCESS_FLAGS>(0u));

		assert(status);
	}

	void workLoopBody() override
	{
		++m_resourceIx;
		if (m_resourceIx >= FRAMES_IN_FLIGHT)
			m_resourceIx = 0;

		auto& commandBuffer = m_commandBuffers[m_resourceIx];
		auto& fence = m_frameComplete[m_resourceIx];

		if (fence)
		{
			while (logicalDevice->waitForFences(1u, &fence.get(), false, MAX_TIMEOUT) == video::IGPUFence::ES_TIMEOUT) {}
			logicalDevice->resetFences(1u, &fence.get());
		}
		else
			fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

		auto renderStart = std::chrono::system_clock::now();
		const auto renderDt = std::chrono::duration_cast<std::chrono::milliseconds>(renderStart - m_lastTime).count();
		m_lastTime = renderStart;
		{ // Calculate Simple Moving Average for FrameTime
			m_time_sum -= m_dtList[m_frame_count];
			m_time_sum += renderDt;
			m_dtList[m_frame_count] = renderDt;
			m_frame_count++;
			if (m_frame_count >= NBL_FRAMES_TO_AVERAGE)
			{
				m_frameDataFilled = true;
				m_frame_count = 0;
			}

		}
		const double averageFrameTime = m_frameDataFilled ? (m_time_sum / (double)NBL_FRAMES_TO_AVERAGE) : (m_time_sum / m_frame_count);

#ifdef NBL_MORE_LOGS
		logger->log("renderDt = %f ------ averageFrameTime = %f", system::ILogger::ELL_INFO, renderDt, averageFrameTime);
#endif // NBL_MORE_LOGS

		auto averageFrameTimeDuration = std::chrono::duration<double, std::milli>(averageFrameTime);
		auto nextPresentationTime = renderStart + averageFrameTimeDuration;
		auto nextPresentationTimeStamp = std::chrono::duration_cast<std::chrono::microseconds>(nextPresentationTime.time_since_epoch());

		inputSystem->getDefaultMouse(&m_mouse);
		inputSystem->getDefaultKeyboard(&m_keyboard);

		m_camera->beginInputProcessing(nextPresentationTimeStamp);
		m_mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { m_camera->mouseProcess(events); }, logger.get());
		m_keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { m_camera->keyboardProcess(events); }, logger.get());
		m_camera->endInputProcessing(nextPresentationTimeStamp);

		const auto& viewMatrix = m_camera->getViewMatrix();
		const auto& viewProjectionMatrix = m_camera->getConcatenatedMatrix();

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

		VkRect2D scissor;
		scissor.offset = { 0, 0 };
		scissor.extent = { WIN_W,WIN_H };
		commandBuffer->setScissor(0u, 1u, &scissor);

		swapchain->acquireNextImage(MAX_TIMEOUT, m_imageAcquire[m_resourceIx].get(), nullptr, &m_acquiredNextFBO);

		video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
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
			beginInfo.framebuffer = fbos[m_acquiredNextFBO];
			beginInfo.renderpass = renderpass;
			beginInfo.renderArea = area;
			beginInfo.clearValues = clear;
		}

		commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);

		for (auto index = 0u; index < m_cpuGpuObjects->objects.size(); ++index)
		{
			const auto iterator = m_cpuGpuObjects->objects[index];
			auto geometryObject = iterator.first;
			auto gpuObject = iterator.second;

			core::matrix3x4SIMD modelMatrix;
			modelMatrix.setTranslation(nbl::core::vectorSIMDf(index * 5, 0, 0, 0));

			core::matrix4SIMD mvp = core::concatenateBFollowedByA(viewProjectionMatrix, modelMatrix);
			auto* gpuGraphicsPipeline = gpuObject.gpuGraphicsPipeline.get();

			commandBuffer->bindGraphicsPipeline(gpuGraphicsPipeline);
			commandBuffer->pushConstants(gpuGraphicsPipeline->getRenderpassIndependentPipeline()->getLayout(), video::IGPUShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), mvp.pointer());
			commandBuffer->drawMeshBuffer(gpuObject.gpuMeshbBuffer.get());
		}

		commandBuffer->endRenderPass();
		commandBuffer->end();

		CommonAPI::Submit(
			logicalDevice.get(),
			swapchain.get(),
			commandBuffer.get(),
			queues[CommonAPI::InitOutput::EQT_GRAPHICS],
			m_imageAcquire[m_resourceIx].get(),
			m_renderFinished[m_resourceIx].get(),
			fence.get());

		CommonAPI::Present(
			logicalDevice.get(),
			swapchain.get(),
			queues[CommonAPI::InitOutput::EQT_GRAPHICS],
			m_renderFinished[m_resourceIx].get(),
			m_acquiredNextFBO);
	}

	bool keepRunning() override
	{
		return windowCb->isWindowOpen();
	}
};

NBL_COMMON_API_MAIN(GeometryCreatorSampleApp)

extern "C" {  _declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; }
