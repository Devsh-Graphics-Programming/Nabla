// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>

#include "../common/Camera.hpp"
#include "../common/CommonAPI.h"

//#include "nbl/ext/ScreenShot/ScreenShot.h"


using namespace nbl;
using namespace core;


#include "nbl/nblpack.h"
struct VertexStruct
{
    /// every member needs to be at location aligned to its type size for GLSL
    float Pos[3]; /// uses float hence need 4 byte alignment
    uint8_t Col[2]; /// same logic needs 1 byte alignment
    uint8_t uselessPadding[2]; /// so if there is a member with 4 byte alignment then whole struct needs 4 byte align, so pad it
} PACK_STRUCT;
#include "nbl/nblunpack.h"

const char* vertexSource = R"===(
#version 430 core

layout(location = 0) in vec4 vPos; //only a 3d position is passed from Nabla, but last (the W) coordinate gets filled with default 1.0
layout(location = 1) in vec4 vCol;

layout( push_constant, row_major ) uniform Block {
	mat4 modelViewProj;
} PushConstants;

layout(location = 0) out vec4 Color; //per vertex output color, will be interpolated across the triangle

void main()
{
    gl_Position = PushConstants.modelViewProj*vPos; //only thing preventing the shader from being core-compliant
    Color = vCol;
}
)===";

const char* fragmentSource = R"===(
#version 430 core

layout(location = 0) in vec4 Color; //per vertex output color, will be interpolated across the triangle

layout(location = 0) out vec4 pixelColor;

void main()
{
    pixelColor = Color;
}
)===";

class GPUMesh : public ApplicationBase
{

public:
	constexpr static uint32_t WIN_W = 1280;
	constexpr static uint32_t WIN_H = 720;
	constexpr static uint32_t SC_IMG_COUNT = 3u;
	constexpr static uint32_t FRAMES_IN_FLIGHT = 5u;
	constexpr static uint64_t MAX_TIMEOUT = 99999999999999ull;
	constexpr static size_t NBL_FRAMES_TO_AVERAGE = 100ull;
	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

	struct Nabla : IUserData
	{
		nbl::core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
		nbl::core::smart_refctd_ptr<nbl::ui::IWindow> window;
		nbl::core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
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
		Camera camera = Camera(vectorSIMDf(0, 0, 0), vectorSIMDf(0, 0, 0), matrix4SIMD());

		int resourceIx = -1;
		uint32_t acquiredNextFBO = {};
		std::chrono::system_clock::time_point lastTime;
		bool frameDataFilled = false;
		size_t frame_count = 0ull;
		double time_sum = 0;
		double dtList[NBL_FRAMES_TO_AVERAGE] = {};

		core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
		core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
		core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
		core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FRAMES_IN_FLIGHT];

		void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override
		{
			window = std::move(wnd);
		}
	};

APP_CONSTRUCTOR(GPUMesh)

	void onAppInitialized_impl(void* data) override
	{
		Nabla* engine = static_cast<Nabla*>(data);

		CommonAPI::InitOutput<SC_IMG_COUNT> initOutput;
		initOutput.window = core::smart_refctd_ptr(engine->window);
		CommonAPI::Init<WIN_W, WIN_H, SC_IMG_COUNT>(initOutput, video::EAT_OPENGL, "GPUMesh", nbl::asset::EF_D32_SFLOAT);
		engine->window = std::move(initOutput.window);
		engine->windowCb = std::move(initOutput.windowCb);
		engine->apiConnection = std::move(initOutput.apiConnection);
		engine->surface = std::move(initOutput.surface);
		engine->utilities = std::move(initOutput.utilities);
		engine->logicalDevice = std::move(initOutput.logicalDevice);
		engine->physicalDevice = initOutput.physicalDevice;
		engine->queues = std::move(initOutput.queues);
		engine->swapchain = std::move(initOutput.swapchain);
		engine->renderpass = std::move(initOutput.renderpass);
		engine->fbos = std::move(initOutput.fbo);
		engine->commandPool = std::move(initOutput.commandPool);
		engine->system = std::move(initOutput.system);
		engine->assetManager = std::move(initOutput.assetManager);
		engine->cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
		engine->logger = std::move(initOutput.logger);
		engine->inputSystem = std::move(initOutput.inputSystem);

		for (size_t i = 0ull; i < NBL_FRAMES_TO_AVERAGE; ++i)
			engine->dtList[i] = 0.0;

		matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60), float(WIN_W) / WIN_H, 0.1, 1000);
		engine->camera = Camera(core::vectorSIMDf(-4, 0, 0), core::vectorSIMDf(0, 0, 0), projectionMatrix);

		engine->logicalDevice->createCommandBuffers(engine->commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, engine->commandBuffers);

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
		{
			engine->imageAcquire[i] = engine->logicalDevice->createSemaphore();
			engine->renderFinished[i] = engine->logicalDevice->createSemaphore();
		}
	}

	void onAppTerminated_impl(void* data) override
	{
		Nabla* engine = static_cast<Nabla*>(data);

		core::rect<uint32_t> sourceRect(0, 0, WIN_W, WIN_H);
		//ext::ScreenShot::dirtyCPUStallingScreenshot(device, "screenshot.png", sourceRect, asset::EF_R8G8B8_SRGB);
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

		const auto& mvp = engine->camera.getConcatenatedMatrix();

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

		//! Stress test for memleaks aside from demo how to create meshes that live on the GPU RAM
		{
			VertexStruct vertices[8];
			vertices[0] = VertexStruct{ {-1.f,-1.f,-1.f},{  0,  0} };
			vertices[1] = VertexStruct{ { 1.f,-1.f,-1.f},{127,  0} };
			vertices[2] = VertexStruct{ {-1.f, 1.f,-1.f},{255,  0} };
			vertices[3] = VertexStruct{ { 1.f, 1.f,-1.f},{  0,127} };
			vertices[4] = VertexStruct{ {-1.f,-1.f, 1.f},{127,127} };
			vertices[5] = VertexStruct{ { 1.f,-1.f, 1.f},{255,127} };
			vertices[6] = VertexStruct{ {-1.f, 1.f, 1.f},{  0,255} };
			vertices[7] = VertexStruct{ { 1.f, 1.f, 1.f},{127,255} };

			uint16_t indices_indexed16[] =
			{
				0,1,2,1,2,3,
				4,5,6,5,6,7,
				0,1,4,1,4,5,
				2,3,6,3,6,7,
				0,2,4,2,4,6,
				1,3,5,3,5,7
			};

		//	auto upStreamBuff = driver->getDefaultUpStreamingBuffer();
		//	core::smart_refctd_ptr<video::IGPUBuffer> upStreamRef(upStreamBuff->getBuffer());

		//	const void* dataToPlace[2] = { vertices,indices_indexed16 };
		//	uint32_t offsets[2] = { video::StreamingTransientDataBufferMT<>::invalid_address,video::StreamingTransientDataBufferMT<>::invalid_address };
		//	uint32_t alignments[2] = { sizeof(decltype(vertices[0u])),sizeof(decltype(indices_indexed16[0u])) };
		//	uint32_t sizes[2] = { sizeof(vertices),sizeof(indices_indexed16) };
		//	upStreamBuff->multi_place(2u, (const void* const*)dataToPlace, (uint32_t*)offsets, (uint32_t*)sizes, (uint32_t*)alignments);
		//	if (upStreamBuff->needsManualFlushOrInvalidate())
		//	{
		//		auto upStreamMem = upStreamBuff->getBuffer()->getBoundMemory();
		//		driver->flushMappedMemoryRanges({ video::IDriverMemoryAllocation::MappedMemoryRange(upStreamMem,offsets[0],sizes[0]),video::IDriverMemoryAllocation::MappedMemoryRange(upStreamMem,offsets[1],sizes[1]) });
		//	}

		//	asset::SPushConstantRange range[1] = { asset::ISpecializedShader::ESS_VERTEX,0u,sizeof(core::matrix4SIMD) };

		//	auto createGPUSpecializedShaderFromSource = [=](const char* source, asset::ISpecializedShader::E_SHADER_STAGE stage)
		//	{
		//		auto spirv = device->getAssetManager()->getGLSLCompiler()->createSPIRVFromGLSL(source, stage, "main", "runtimeID");
		//		auto unspec = driver->createGPUShader(std::move(spirv));
		//		return driver->createGPUSpecializedShader(unspec.get(), { nullptr,nullptr,"main",stage });
		//	};
		//	// origFilepath is only relevant when you have filesystem #includes in your shader
		//	auto createGPUSpecializedShaderFromSourceWithIncludes = [&](const char* source, asset::ISpecializedShader::E_SHADER_STAGE stage, const char* origFilepath)
		//	{
		//		auto resolved_includes = device->getAssetManager()->getGLSLCompiler()->resolveIncludeDirectives(source, stage, origFilepath);
		//		return createGPUSpecializedShaderFromSource(reinterpret_cast<const char*>(resolved_includes->getSPVorGLSL()->getPointer()), stage);
		//	};
		//	core::smart_refctd_ptr<video::IGPUSpecializedShader> shaders[2] =
		//	{
		//		createGPUSpecializedShaderFromSourceWithIncludes(vertexSource,asset::ISpecializedShader::ESS_VERTEX, "shader.vert"),
		//		createGPUSpecializedShaderFromSource(fragmentSource,asset::ISpecializedShader::ESS_FRAGMENT)
		//	};
		//	auto shadersPtr = reinterpret_cast<video::IGPUSpecializedShader**>(shaders);

		//	asset::SVertexInputParams inputParams;
		//	inputParams.enabledAttribFlags = 0b11u;
		//	inputParams.enabledBindingFlags = 0b1u;
		//	inputParams.attributes[0].binding = 0u;
		//	inputParams.attributes[0].format = asset::EF_R32G32B32_SFLOAT;
		//	inputParams.attributes[0].relativeOffset = offsetof(VertexStruct, Pos[0]);
		//	inputParams.attributes[1].binding = 0u;
		//	inputParams.attributes[1].format = asset::EF_R8G8_UNORM;
		//	inputParams.attributes[1].relativeOffset = offsetof(VertexStruct, Col[0]);
		//	inputParams.bindings[0].stride = sizeof(VertexStruct);
		//	inputParams.bindings[0].inputRate = asset::EVIR_PER_VERTEX;

		//	asset::SBlendParams blendParams; // defaults are sane

		//	asset::SPrimitiveAssemblyParams assemblyParams = { asset::EPT_TRIANGLE_LIST,false,1u };

		//	asset::SStencilOpParams defaultStencil;
		//	asset::SRasterizationParams rasterParams;
		//	rasterParams.faceCullingMode = asset::EFCM_NONE;
		//	auto pipeline = driver->createGPURenderpassIndependentPipeline(nullptr, driver->createGPUPipelineLayout(range, range + 1u, nullptr, nullptr, nullptr, nullptr),
		//		shadersPtr, shadersPtr + sizeof(shaders) / sizeof(core::smart_refctd_ptr<video::IGPUSpecializedShader>),
		//		inputParams, blendParams, assemblyParams, rasterParams);

		//	asset::SBufferBinding<video::IGPUBuffer> bindings[video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
		//	bindings[0u] = { offsets[0],upStreamRef };
		//	auto mb = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(std::move(pipeline), nullptr, bindings, asset::SBufferBinding<video::IGPUBuffer>{offsets[1], upStreamRef});
		//	{
		//		mb->setIndexType(asset::EIT_16BIT);
		//		mb->setIndexCount(2 * 3 * 6);
		//	}

		//	driver->bindGraphicsPipeline(mb->getPipeline());
		//	driver->pushConstants(mb->getPipeline()->getLayout(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), mvp.pointer());
		//	driver->drawMeshBuffer(mb.get());

		//	upStreamBuff->multi_free(2u, (uint32_t*)&offsets, (uint32_t*)&sizes, driver->placeFence());
		//}
		//driver->endScene();
	}

	bool keepRunning(void* params) override
	{
		Nabla* engine = static_cast<Nabla*>(params);
		return engine->windowCb->isWindowOpen();
	}
};

NBL_COMMON_API_MAIN(GPUMesh, GPUMesh::Nabla)