// Copyright (C) 2018-2021 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>

#include "../common/QToQuitEventReceiver.h"
#include "../common/Camera.hpp"
#include "../common/CommonAPI.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"

using namespace nbl;
using namespace core;

/*
	Uncomment for more detailed logging
*/

// #define NBL_MORE_LOGS

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

int main()
{
	constexpr uint32_t WIN_W = 1280;
	constexpr uint32_t WIN_H = 720;
	constexpr uint32_t FBO_COUNT = 1u;

	auto initOutput = CommonAPI::Init<WIN_W, WIN_H, FBO_COUNT>(video::EAT_OPENGL, "GeometryCreator", nbl::asset::EF_D32_SFLOAT);
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
	auto cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
	nbl::video::IGPUObjectFromAssetConverter cpu2gpu;

	core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> commandBuffer;
	logicalDevice->createCommandBuffers(commandPool.get(), nbl::video::IGPUCommandBuffer::EL_PRIMARY, 1, &commandBuffer);

	auto geometryCreator = assetManager->getGeometryCreator();
	auto cubeGeometry = geometryCreator->createCubeMesh(vector3df(2,2,2));
	auto sphereGeometry = geometryCreator->createSphereMesh(2, 16, 16);
	auto cylinderGeometry = geometryCreator->createCylinderMesh(2, 2, 20);
	auto rectangleGeometry = geometryCreator->createRectangleMesh(nbl::core::vector2df_SIMD(1.5, 3));
	auto diskGeometry = geometryCreator->createDiskMesh(2, 30);
	auto coneGeometry = geometryCreator->createConeMesh(2, 3, 10);
	auto arrowGeometry = geometryCreator->createArrowMesh();
	auto icosphereGeometry = geometryCreator->createIcoSphere(1, 3, true);

	auto createGPUSpecializedShaderFromSource = [=](const char* source, asset::ISpecializedShader::E_SHADER_STAGE stage) -> core::smart_refctd_ptr<video::IGPUSpecializedShader>
	{
		auto spirv = assetManager->getGLSLCompiler()->createSPIRVFromGLSL(source, stage, "main", "runtimeID");
		if (!spirv)
			return nullptr;

		auto gpuUnspecializedShader = logicalDevice->createGPUShader(std::move(spirv));
		return logicalDevice->createGPUSpecializedShader(gpuUnspecializedShader.get(), { nullptr, nullptr, "main", stage });
	};

	auto createGPUSpecializedShaderFromSourceWithIncludes = [&](const char* source, asset::ISpecializedShader::E_SHADER_STAGE stage, const char* origFilepath)
	{
		auto resolved_includes = assetManager->getGLSLCompiler()->resolveIncludeDirectives(source, stage, origFilepath);
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
		auto gpuRenderpassIndependentPipeline = logicalDevice->createGPURenderpassIndependentPipeline
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

		auto gpubuffers = cpu2gpu.getGPUObjectsFromAssets(cpubuffers.data(), cpubuffers.data() + 1, cpu2gpuParams);
		{
			auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(cpubuffers.data(), cpubuffers.data() + 1, cpu2gpuParams);
			if (!gpubuffers || gpubuffers->size() < 1u)
				assert(false);
		}

		asset::SBufferBinding<video::IGPUBuffer> bindings[MAX_DATA_BUFFERS];
		for (auto i=0,j=0; i < MAX_ATTR_BUF_BINDING_COUNT; i++)
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

	auto gpuCube = createGPUMeshBufferAndItsPipeline(cubeGeometry);
	auto gpuSphere = createGPUMeshBufferAndItsPipeline(sphereGeometry);
	auto gpuCylinder = createGPUMeshBufferAndItsPipeline(cylinderGeometry);
	auto gpuRectangle = createGPUMeshBufferAndItsPipeline(rectangleGeometry);
	auto gpuDisk = createGPUMeshBufferAndItsPipeline(diskGeometry);
	auto gpuCone = createGPUMeshBufferAndItsPipeline(coneGeometry);
	auto gpuArrow = createGPUMeshBufferAndItsPipeline(arrowGeometry);
	auto gpuIcosphere = createGPUMeshBufferAndItsPipeline(icosphereGeometry);

	Objects cpuGpuObjects =
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
	
	QToQuitEventReceiver escaper;
	CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
	CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

	core::vectorSIMDf cameraPosition(0, 5, -10);
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
	while(escaper.keepOpen())
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
		keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); escaper.process(events); }, logger.get());
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
		asset::SClearValue clear[2] = {};
		clear[0].color.float32[0] = 1.f;
		clear[0].color.float32[1] = 1.f;
		clear[0].color.float32[2] = 1.f;
		clear[0].color.float32[3] = 1.f;
		clear[1].depthStencil.depth = 0.f;

		beginInfo.clearValueCount = 2u;
		beginInfo.framebuffer = fbo;
		beginInfo.renderpass = renderpass;
		beginInfo.renderArea = area;
		beginInfo.clearValues = clear;

		commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);

		for (auto index = 0u; index < cpuGpuObjects.objects.size(); ++index)
		{
			const auto iterator = cpuGpuObjects.objects[index];
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
