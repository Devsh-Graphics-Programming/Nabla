// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>

#include "../common/Camera.hpp"
#include "../common/CommonAPI.h"

/*
	General namespaces. Entire engine consists of those bellow.
*/

using namespace nbl;
using namespace asset;
using namespace video;
using namespace core;

/*
	Uncomment for more detailed logging
*/

// #define NBL_MORE_LOGS

class NablaTutorialExampleApp : public ApplicationBase
{
	/*
		 SIrrlichtCreationParameters holds some specific initialization information
		 about driver being used, size of window, stencil buffer or depth buffer.
		 Used to create a device.
	*/

	constexpr static uint32_t WIN_W = 1280;
	constexpr static uint32_t WIN_H = 720;
	constexpr static uint32_t SC_IMG_COUNT = 3u;
	constexpr static uint32_t FRAMES_IN_FLIGHT = 5u;
	constexpr static uint64_t MAX_TIMEOUT = 99999999999999ull;
	constexpr static size_t NBL_FRAMES_TO_AVERAGE = 100ull;
	
	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

public:
	struct Nabla : IUserData
	{
	/*
		Most important objects to manage literally whole stuff are bellow.
		By their usage you can create for example GPU objects, load or write
		assets or manage objects on a scene.
	*/

		nbl::core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
		nbl::core::smart_refctd_ptr<nbl::ui::IWindow> window;
		nbl::core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
		nbl::core::smart_refctd_ptr<nbl::video::IAPIConnection> apiConnection;
		nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface;
		nbl::core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
		nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
		nbl::video::IPhysicalDevice* physicalDevice;
		std::array<nbl::video::IGPUQueue*, CommonAPI::InitOutput<SC_IMG_COUNT>::EQT_COUNT> queues = { nullptr, nullptr, nullptr, nullptr };
		nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
		std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, SC_IMG_COUNT> fbos;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool> commandPool; // TODO: Multibuffer and reset the commandpools
		nbl::core::smart_refctd_ptr<nbl::system::ISystem> system;
		nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
		nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
		nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger;
		nbl::core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;

		nbl::core::smart_refctd_ptr<video::IGPUFence> gpuTransferFence;
		nbl::core::smart_refctd_ptr<video::IGPUFence> gpuComputeFence;
		nbl::video::IGPUObjectFromAssetConverter cpu2gpu;

		core::smart_refctd_ptr<video::IGPUMeshBuffer> gpuMeshBuffer;
		core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> gpuRenderpassIndependentPipeline;
		core::smart_refctd_ptr<IGPUBuffer> gpuubo;
		core::smart_refctd_ptr<IGPUDescriptorSet> gpuDescriptorSet1;
		core::smart_refctd_ptr<IGPUDescriptorSet> gpuDescriptorSet3;
		core::smart_refctd_ptr<IGPUGraphicsPipeline> gpuGraphicsPipeline;

		core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
		core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
		core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
		core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FRAMES_IN_FLIGHT];

		CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
		CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;
		Camera camera = Camera(vectorSIMDf(0, 0, 0), vectorSIMDf(0, 0, 0), matrix4SIMD());

		uint32_t ds1UboBinding = 0;
		int resourceIx;
		uint32_t acquiredNextFBO = {};
		std::chrono::system_clock::time_point lastTime;
		bool frameDataFilled = false;
		size_t frame_count = 0ull;
		double time_sum = 0;
		double dtList[NBL_FRAMES_TO_AVERAGE] = {};

		auto createDescriptorPool(const uint32_t textureCount)
		{
			constexpr uint32_t maxItemCount = 256u;
			{
				nbl::video::IDescriptorPool::SDescriptorPoolSize poolSize;
				poolSize.count = textureCount;
				poolSize.type = nbl::asset::EDT_COMBINED_IMAGE_SAMPLER;
				return logicalDevice->createDescriptorPool(static_cast<nbl::video::IDescriptorPool::E_CREATE_FLAGS>(0), maxItemCount, 1u, &poolSize);
			}
		}

		void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override
		{
			window = std::move(wnd);
		}
	};

	APP_CONSTRUCTOR(NablaTutorialExampleApp)

	void onAppInitialized_impl(void* data) override
	{
		Nabla* engine = static_cast<Nabla*>(data);
		CommonAPI::InitOutput<SC_IMG_COUNT> initOutput;
		initOutput.window = core::smart_refctd_ptr(engine->window);
		CommonAPI::Init<WIN_W, WIN_H, SC_IMG_COUNT>(initOutput, video::EAT_OPENGL, "NablaTutorialExample", nbl::asset::EF_D32_SFLOAT);
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

		engine->gpuTransferFence = engine->logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
		engine->gpuComputeFence = engine->logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

		nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
		{
			engine->cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].fence = &engine->gpuTransferFence;
			engine->cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].fence = &engine->gpuComputeFence;
		}

		/*
			Helpfull class for managing basic geometry objects.
			Thanks to it you can get half filled pipeline for your
			geometries such as cubes, cones or spheres.
		*/

		auto geometryCreator = engine->assetManager->getGeometryCreator();
		auto rectangleGeometry = geometryCreator->createRectangleMesh(nbl::core::vector2df_SIMD(1.5, 3));

		/*
		Loading an asset bundle. You can specify some flags
		and parameters to have an impact on extraordinary
		tasks while loading for example.
	*/

		asset::IAssetLoader::SAssetLoadParams loadingParams;
		auto images_bundle = engine->assetManager->getAsset("../../media/color_space_test/R8G8B8A8_1.png", loadingParams);
		assert(!images_bundle.getContents().empty());
		auto image = images_bundle.getContents().begin()[0];
		auto image_raw = static_cast<asset::ICPUImage*>(image.get());

		/*
			Specifing gpu image view parameters to create a gpu
			image view through the driver.
		*/

		auto gpuImage = cpu2gpu.getGPUObjectsFromAssets(&image_raw, &image_raw + 1, engine->cpu2gpuParams)->front();
		engine->cpu2gpuParams.waitForCreationToComplete();
		auto& gpuParams = gpuImage->getCreationParameters();

		IImageView<IGPUImage>::SCreationParams gpuImageViewParams = { static_cast<IGPUImageView::E_CREATE_FLAGS>(0), gpuImage, IImageView<IGPUImage>::ET_2D, gpuParams.format, {}, {static_cast<IImage::E_ASPECT_FLAGS>(0u), 0, gpuParams.mipLevels, 0, gpuParams.arrayLayers} };
		auto gpuImageView = engine->logicalDevice->createGPUImageView(std::move(gpuImageViewParams));

		/*
			Specifying cache key to default exsisting cached asset bundle
			and specifying it's size where end is determined by
			static_cast<IAsset::E_TYPE>(0u)
		*/

		const IAsset::E_TYPE types[]{ IAsset::E_TYPE::ET_SPECIALIZED_SHADER, IAsset::E_TYPE::ET_SPECIALIZED_SHADER, static_cast<IAsset::E_TYPE>(0u) };

		auto cpuVertexShader = core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(engine->assetManager->findAssets("nbl/builtin/material/lambertian/singletexture/specialized_shader.vert", types)->front().getContents().begin()[0]);
		auto cpuFragmentShader = core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(engine->assetManager->findAssets("nbl/builtin/material/lambertian/singletexture/specialized_shader.frag", types)->front().getContents().begin()[0]);

		auto gpuVertexShader = cpu2gpu.getGPUObjectsFromAssets(&cpuVertexShader.get(), &cpuVertexShader.get() + 1, engine->cpu2gpuParams)->front();
		engine->cpu2gpuParams.waitForCreationToComplete();
		auto gpuFragmentShader = cpu2gpu.getGPUObjectsFromAssets(&cpuFragmentShader.get(), &cpuFragmentShader.get() + 1, engine->cpu2gpuParams)->front();
		engine->cpu2gpuParams.waitForCreationToComplete();
		engine->cpu2gpuParams.waitForCreationToComplete();
		std::array<IGPUSpecializedShader*, 2> gpuShaders = { gpuVertexShader.get(), gpuFragmentShader.get() };

		size_t ds0SamplerBinding = 0, ds1UboBinding = 0;
		auto createAndGetUsefullData = [&](asset::IGeometryCreator::return_type& geometryObject)
		{
			/*
				SBinding for the texture (sampler).
			*/

			IGPUDescriptorSetLayout::SBinding gpuSamplerBinding;
			gpuSamplerBinding.binding = ds0SamplerBinding;
			gpuSamplerBinding.type = EDT_COMBINED_IMAGE_SAMPLER;
			gpuSamplerBinding.count = 1u;
			gpuSamplerBinding.stageFlags = static_cast<IGPUSpecializedShader::E_SHADER_STAGE>(IGPUSpecializedShader::ESS_FRAGMENT);
			gpuSamplerBinding.samplers = nullptr;

			/*
				SBinding for UBO - basic view parameters.
			*/

			IGPUDescriptorSetLayout::SBinding gpuUboBinding;
			gpuUboBinding.count = 1u;
			gpuUboBinding.binding = ds1UboBinding;
			gpuUboBinding.stageFlags = static_cast<asset::ICPUSpecializedShader::E_SHADER_STAGE>(asset::ICPUSpecializedShader::ESS_VERTEX | asset::ICPUSpecializedShader::ESS_FRAGMENT);
			gpuUboBinding.type = asset::EDT_UNIFORM_BUFFER;

			/*
				Creating specific descriptor set layouts from specialized bindings.
				Those layouts needs to attached to pipeline layout if required by user.
				IrrlichtBaW provides 4 places for descriptor set layout usage.
			*/

			auto gpuDs1Layout = engine->logicalDevice->createGPUDescriptorSetLayout(&gpuUboBinding, &gpuUboBinding + 1);
			auto gpuDs3Layout = engine->logicalDevice->createGPUDescriptorSetLayout(&gpuSamplerBinding, &gpuSamplerBinding + 1);

			/*
				Creating gpu UBO with appropiate size.

				We know ahead of time that `SBasicViewParameters` struct is the expected structure of the only UBO block in the descriptor set nr. 1 of the shader.
			*/

			IGPUBuffer::SCreationParams creationParams;
			creationParams.usage = asset::IBuffer::EUF_UNIFORM_BUFFER_BIT;
			IDriverMemoryBacked::SDriverMemoryRequirements memReq;
			memReq.vulkanReqs.size = sizeof(SBasicViewParameters);
			auto gpuubo = engine->logicalDevice->createGPUBufferOnDedMem(creationParams, memReq, true);

			/*
				Creating descriptor sets - texture (sampler) and basic view parameters (UBO).
				Specifying info and write parameters for updating certain descriptor set to the driver.

				We know ahead of time that `SBasicViewParameters` struct is the expected structure of the only UBO block in the descriptor set nr. 1 of the shader.
			*/

			auto descriptorPool = engine->createDescriptorPool(1u);

			auto gpuDescriptorSet3 = engine->logicalDevice->createGPUDescriptorSet(descriptorPool.get(), gpuDs3Layout);
			{
				video::IGPUDescriptorSet::SWriteDescriptorSet write;
				write.dstSet = gpuDescriptorSet3.get();
				write.binding = ds0SamplerBinding;
				write.count = 1u;
				write.arrayElement = 0u;
				write.descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
				IGPUDescriptorSet::SDescriptorInfo info;
				{
					info.desc = std::move(gpuImageView);
					ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE,ISampler::ETC_CLAMP_TO_EDGE,ISampler::ETC_CLAMP_TO_EDGE,ISampler::ETBC_FLOAT_OPAQUE_BLACK,ISampler::ETF_LINEAR,ISampler::ETF_LINEAR,ISampler::ESMM_LINEAR,0u,false,ECO_ALWAYS };
					info.image = { engine->logicalDevice->createGPUSampler(samplerParams),EIL_SHADER_READ_ONLY_OPTIMAL };
				}
				write.info = &info;
				engine->logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);
			}

			auto gpuDescriptorSet1 = engine->logicalDevice->createGPUDescriptorSet(descriptorPool.get(), gpuDs1Layout);
			{
				video::IGPUDescriptorSet::SWriteDescriptorSet write;
				write.dstSet = gpuDescriptorSet1.get();
				write.binding = ds1UboBinding;
				write.count = 1u;
				write.arrayElement = 0u;
				write.descriptorType = asset::EDT_UNIFORM_BUFFER;
				video::IGPUDescriptorSet::SDescriptorInfo info;
				{
					info.desc = gpuubo;
					info.buffer.offset = 0ull;
					info.buffer.size = sizeof(SBasicViewParameters);
				}
				write.info = &info;
				engine->logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);
			}

			auto gpuPipelineLayout = engine->logicalDevice->createGPUPipelineLayout(nullptr, nullptr, nullptr, std::move(gpuDs1Layout), nullptr, std::move(gpuDs3Layout));

			/*
				Preparing required pipeline parameters and filling choosen one.
				Note that some of them are returned from geometry creator according
				to what I mentioned in returning half pipeline parameters.
			*/

			asset::SBlendParams blendParams;
			asset::SRasterizationParams rasterParams;
			rasterParams.faceCullingMode = asset::EFCM_NONE;

			/*
				Creating gpu pipeline with it's pipeline layout and specilized parameters.
				Attaching vertex shader and fragment shaders.
			*/

			auto gpuPipeline = engine->logicalDevice->createGPURenderpassIndependentPipeline(nullptr, std::move(gpuPipelineLayout), gpuShaders.data(), gpuShaders.data() + gpuShaders.size(), geometryObject.inputParams, blendParams, geometryObject.assemblyParams, rasterParams);

			nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
			graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(gpuPipeline.get());
			graphicsPipelineParams.renderpass = core::smart_refctd_ptr(engine->renderpass);
			auto gpuGraphicsPipeline = engine->logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));

			core::vectorSIMDf cameraPosition(-5, 0, 0);
			matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60), float(WIN_W) / WIN_H, 0.01, 1000);
			engine->camera = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), projectionMatrix, 10.f, 1.f);

			/*
				Creating gpu meshbuffer from parameters fetched from geometry creator return value.
			*/

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
			engine->cpu2gpuParams.waitForCreationToComplete();

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

			auto mb = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(core::smart_refctd_ptr(gpuPipeline), nullptr, bindings, std::move(bindings[MAX_ATTR_BUF_BINDING_COUNT]));
			{
				mb->setIndexType(geometryObject.indexType);
				mb->setIndexCount(geometryObject.indexCount);
				mb->setBoundingBox(geometryObject.bbox);
			}

			return std::make_tuple(mb, gpuPipeline, gpuubo, gpuDescriptorSet1, gpuDescriptorSet3, gpuGraphicsPipeline);
		};

		auto gpuRectangle = createAndGetUsefullData(rectangleGeometry);
		engine->gpuMeshBuffer = std::get<0>(gpuRectangle);
		engine->gpuRenderpassIndependentPipeline = std::get<1>(gpuRectangle);
		engine->gpuubo = std::get<2>(gpuRectangle);
		engine->gpuDescriptorSet1 = std::get<3>(gpuRectangle);
		engine->gpuDescriptorSet3 = std::get<4>(gpuRectangle);
		engine->gpuGraphicsPipeline = std::get<5>(gpuRectangle);

		engine->logicalDevice->createCommandBuffers(engine->commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, engine->commandBuffers);

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
		{
			engine->imageAcquire[i] = engine->logicalDevice->createSemaphore();
			engine->renderFinished[i] = engine->logicalDevice->createSemaphore();
		}
	}

	/*
		Hot loop for rendering a scene.
	*/

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
			clear[0].color.float32[0] = 0.f;
			clear[0].color.float32[1] = 0.f;
			clear[0].color.float32[2] = 0.f;
			clear[0].color.float32[3] = 1.f;
			clear[1].depthStencil.depth = 0.f;

			beginInfo.clearValueCount = 2u;
			beginInfo.framebuffer = engine->fbos[engine->acquiredNextFBO];
			beginInfo.renderpass = engine->renderpass;
			beginInfo.renderArea = area;
			beginInfo.clearValues = clear;
		}

		commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);

		const auto viewProjection = engine->camera.getConcatenatedMatrix();
		core::matrix3x4SIMD modelMatrix;
		modelMatrix.setRotation(nbl::core::quaternion(0, 1, 0));

		auto mv = core::concatenateBFollowedByA(engine->camera.getViewMatrix(), modelMatrix);
		auto mvp = core::concatenateBFollowedByA(viewProjection, modelMatrix);
		core::matrix3x4SIMD normalMat;
		mv.getSub3x3InverseTranspose(normalMat);

		/*
			Updating UBO for basic view parameters and sending
			updated data to staging buffer that will redirect
			the data to graphics card - to vertex shader.
		*/

		SBasicViewParameters uboData;
		memcpy(uboData.MV, mv.pointer(), sizeof(mv));
		memcpy(uboData.MVP, mvp.pointer(), sizeof(mvp));
		memcpy(uboData.NormalMat, normalMat.pointer(), sizeof(normalMat));
		commandBuffer->updateBuffer(engine->gpuubo.get(), 0ull, engine->gpuubo->getSize(), &uboData);

		/*
			Binding the most important objects needed to
			render anything on the screen with textures:

			- gpu pipeline
			- gpu descriptor sets
		*/

		commandBuffer->bindGraphicsPipeline(engine->gpuGraphicsPipeline.get());
		commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, engine->gpuRenderpassIndependentPipeline->getLayout(), 1u, 1u, &engine->gpuDescriptorSet1.get(), nullptr);
		commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, engine->gpuRenderpassIndependentPipeline->getLayout(), 3u, 1u, &engine->gpuDescriptorSet3.get(), nullptr);

		/*
			Drawing a mesh (created rectangle) with it's gpu mesh buffer usage.
		*/

		commandBuffer->drawMeshBuffer(engine->gpuMeshBuffer.get());

		commandBuffer->endRenderPass();
		commandBuffer->end();

		CommonAPI::Submit(engine->logicalDevice.get(), engine->swapchain.get(), commandBuffer.get(), engine->queues[CommonAPI::InitOutput<1>::EQT_GRAPHICS], engine->imageAcquire[engine->resourceIx].get(), engine->renderFinished[engine->resourceIx].get(), fence.get());
		CommonAPI::Present(engine->logicalDevice.get(), engine->swapchain.get(), engine->queues[CommonAPI::InitOutput<1>::EQT_GRAPHICS], engine->renderFinished[engine->resourceIx].get(), engine->acquiredNextFBO);
	}

	bool keepRunning(void* params) override
	{
		Nabla* engine = static_cast<Nabla*>(params);
		return engine->windowCb->isWindowOpen();
	}
};

NBL_COMMON_API_MAIN(NablaTutorialExampleApp, NablaTutorialExampleApp::Nabla)
