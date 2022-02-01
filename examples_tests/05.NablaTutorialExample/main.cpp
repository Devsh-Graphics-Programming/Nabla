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
	std::array<nbl::video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> queues = { nullptr, nullptr, nullptr, nullptr };
	nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
	nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, CommonAPI::InitOutput::MaxSwapChainImageCount> fbo;
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxQueuesCount> commandPools; // TODO: Multibuffer and reset the commandpools
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
	
	CommonAPI::InputSystem::ChannelReader<ui::IMouseEventChannel> mouse;
	CommonAPI::InputSystem::ChannelReader<ui::IKeyboardEventChannel> keyboard;
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
	void setSystem(core::smart_refctd_ptr<nbl::system::ISystem>&& s) override
	{
		system = std::move(s);
	}
	nbl::ui::IWindow* getWindow() override
	{
		return window.get();
	}
	video::IAPIConnection* getAPIConnection() override
	{
		return apiConnection.get();
	}
	video::ILogicalDevice* getLogicalDevice()  override
	{
		return logicalDevice.get();
	}
	video::IGPURenderpass* getRenderpass() override
	{
		return renderpass.get();
	}
	void setSurface(core::smart_refctd_ptr<video::ISurface>&& s) override
	{
		surface = std::move(s);
	}
	void setFBOs(std::vector<core::smart_refctd_ptr<video::IGPUFramebuffer>>& f) override
	{
		for (int i = 0; i < f.size(); i++)
		{
			fbo[i] = core::smart_refctd_ptr(f[i]);
		}
	}
	void setSwapchain(core::smart_refctd_ptr<video::ISwapchain>&& s) override
	{
		swapchain = std::move(s);
	}
	uint32_t getSwapchainImageCount() override
	{
		return SC_IMG_COUNT;
	}
	virtual nbl::asset::E_FORMAT getDepthFormat() override
	{
		return nbl::asset::EF_D32_SFLOAT;
	}

	APP_CONSTRUCTOR(NablaTutorialExampleApp)

	void onAppInitialized_impl() override
	{
		CommonAPI::InitOutput initOutput;
        initOutput.window = core::smart_refctd_ptr(window);
        initOutput.system = core::smart_refctd_ptr(system);

		const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT);
		const video::ISurface::SFormat surfaceFormat(asset::EF_R8G8B8A8_SRGB, asset::ECP_COUNT, asset::EOTF_UNKNOWN);

        CommonAPI::InitWithDefaultExt(initOutput, video::EAT_OPENGL_ES, "NablaTutorialExample", WIN_W, WIN_H, SC_IMG_COUNT, swapchainImageUsage, surfaceFormat, nbl::asset::EF_D32_SFLOAT);
		window = std::move(initOutput.window);
		windowCb = std::move(initOutput.windowCb);
		apiConnection = std::move(initOutput.apiConnection);
		surface = std::move(initOutput.surface);
		utilities = std::move(initOutput.utilities);
		logicalDevice = std::move(initOutput.logicalDevice);
		physicalDevice = initOutput.physicalDevice;
		queues = std::move(initOutput.queues);
		swapchain = std::move(initOutput.swapchain);
		renderpass = std::move(initOutput.renderpass);
		fbo = std::move(initOutput.fbo);
		commandPools = std::move(initOutput.commandPools);
		system = std::move(initOutput.system);
		assetManager = std::move(initOutput.assetManager);
		cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
		logger = std::move(initOutput.logger);
		inputSystem = std::move(initOutput.inputSystem);

		gpuTransferFence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
		gpuComputeFence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

		/*
			Helpfull class for managing basic geometry objects.
			Thanks to it you can get half filled pipeline for your
			geometries such as cubes, cones or spheres.
		*/

		auto geometryCreator = assetManager->getGeometryCreator();
		auto rectangleGeometry = geometryCreator->createRectangleMesh(nbl::core::vector2df_SIMD(1.5, 3));

		/*
		Loading an asset bundle. You can specify some flags
		and parameters to have an impact on extraordinary
		tasks while loading for example.
	*/

		asset::IAssetLoader::SAssetLoadParams loadingParams;
		auto images_bundle = assetManager->getAsset("../../media/color_space_test/R8G8B8A8_1.png", loadingParams);
		assert(!images_bundle.getContents().empty());
		auto image = images_bundle.getContents().begin()[0];
		auto image_raw = static_cast<asset::ICPUImage*>(image.get());

		/*
			Specifing gpu image view parameters to create a gpu
			image view through the driver.
		*/

		cpu2gpuParams.beginCommandBuffers();
		auto gpuImage = cpu2gpu.getGPUObjectsFromAssets(&image_raw, &image_raw + 1, cpu2gpuParams)->front();
		cpu2gpuParams.waitForCreationToComplete();
		auto& gpuParams = gpuImage->getCreationParameters();

		IImageView<IGPUImage>::SCreationParams gpuImageViewParams = { static_cast<IGPUImageView::E_CREATE_FLAGS>(0), gpuImage, IImageView<IGPUImage>::ET_2D, gpuParams.format, {}, {static_cast<IImage::E_ASPECT_FLAGS>(0u), 0, gpuParams.mipLevels, 0, gpuParams.arrayLayers} };
		auto gpuImageView = logicalDevice->createGPUImageView(std::move(gpuImageViewParams));

		/*
			Specifying cache key to default exsisting cached asset bundle
			and specifying it's size where end is determined by
			static_cast<IAsset::E_TYPE>(0u)
		*/

		const IAsset::E_TYPE types[]{ IAsset::E_TYPE::ET_SPECIALIZED_SHADER, IAsset::E_TYPE::ET_SPECIALIZED_SHADER, static_cast<IAsset::E_TYPE>(0u) };

		auto cpuVertexShader = core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(assetManager->findAssets("nbl/builtin/material/lambertian/singletexture/specialized_shader.vert", types)->front().getContents().begin()[0]);
		auto cpuFragmentShader = core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(assetManager->findAssets("nbl/builtin/material/lambertian/singletexture/specialized_shader.frag", types)->front().getContents().begin()[0]);

		cpu2gpuParams.beginCommandBuffers();
		auto gpuVertexShader = cpu2gpu.getGPUObjectsFromAssets(&cpuVertexShader.get(), &cpuVertexShader.get() + 1, cpu2gpuParams)->front();
		auto gpuFragmentShader = cpu2gpu.getGPUObjectsFromAssets(&cpuFragmentShader.get(), &cpuFragmentShader.get() + 1, cpu2gpuParams)->front();
		cpu2gpuParams.waitForCreationToComplete();
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
			gpuSamplerBinding.stageFlags = static_cast<IGPUShader::E_SHADER_STAGE>(IGPUShader::ESS_FRAGMENT);
			gpuSamplerBinding.samplers = nullptr;

			/*
				SBinding for UBO - basic view parameters.
			*/

			IGPUDescriptorSetLayout::SBinding gpuUboBinding;
			gpuUboBinding.count = 1u;
			gpuUboBinding.binding = ds1UboBinding;
			gpuUboBinding.stageFlags = static_cast<asset::ICPUShader::E_SHADER_STAGE>(asset::ICPUShader::ESS_VERTEX | asset::ICPUShader::ESS_FRAGMENT);
			gpuUboBinding.type = asset::EDT_UNIFORM_BUFFER;

			/*
				Creating specific descriptor set layouts from specialized bindings.
				Those layouts needs to attached to pipeline layout if required by user.
				IrrlichtBaW provides 4 places for descriptor set layout usage.
			*/

			auto gpuDs1Layout = logicalDevice->createGPUDescriptorSetLayout(&gpuUboBinding, &gpuUboBinding + 1);
			auto gpuDs3Layout = logicalDevice->createGPUDescriptorSetLayout(&gpuSamplerBinding, &gpuSamplerBinding + 1);

			/*
				Creating gpu UBO with appropiate size.

				We know ahead of time that `SBasicViewParameters` struct is the expected structure of the only UBO block in the descriptor set nr. 1 of the shader.
			*/

			IGPUBuffer::SCreationParams creationParams;
			creationParams.canUpdateSubRange = true;
			creationParams.usage = asset::IBuffer::EUF_UNIFORM_BUFFER_BIT;
			creationParams.sharingMode = asset::E_SHARING_MODE::ESM_CONCURRENT;
			creationParams.queueFamilyIndexCount = 0u;
			creationParams.queueFamilyIndices = nullptr;
			IDriverMemoryBacked::SDriverMemoryRequirements memReq;
			memReq.vulkanReqs.size = sizeof(SBasicViewParameters);
			gpuubo = logicalDevice->createGPUBufferOnDedMem(creationParams, memReq);

			/*
				Creating descriptor sets - texture (sampler) and basic view parameters (UBO).
				Specifying info and write parameters for updating certain descriptor set to the driver.

				We know ahead of time that `SBasicViewParameters` struct is the expected structure of the only UBO block in the descriptor set nr. 1 of the shader.
			*/

			auto descriptorPool = createDescriptorPool(1u);

			auto gpuDescriptorSet3 = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), gpuDs3Layout);
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
					info.image = { logicalDevice->createGPUSampler(samplerParams),EIL_SHADER_READ_ONLY_OPTIMAL };
				}
				write.info = &info;
				logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);
			}

			auto gpuDescriptorSet1 = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), gpuDs1Layout);
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
				logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);
			}

			auto gpuPipelineLayout = logicalDevice->createGPUPipelineLayout(nullptr, nullptr, nullptr, std::move(gpuDs1Layout), nullptr, std::move(gpuDs3Layout));

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

			auto gpuPipeline = logicalDevice->createGPURenderpassIndependentPipeline(nullptr, std::move(gpuPipelineLayout), gpuShaders.data(), gpuShaders.data() + gpuShaders.size(), geometryObject.inputParams, blendParams, geometryObject.assemblyParams, rasterParams);

			nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
			graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(gpuPipeline.get());
			graphicsPipelineParams.renderpass = core::smart_refctd_ptr(renderpass);
			auto gpuGraphicsPipeline = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));

			core::vectorSIMDf cameraPosition(-5, 0, 0);
			matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60.0f), float(WIN_W) / WIN_H, 0.01, 1000);
			camera = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), projectionMatrix, 10.f, 1.f);

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

			auto gpubuffers = cpu2gpu.getGPUObjectsFromAssets(cpubuffers.data(), cpubuffers.data() + cpubuffers.size(), cpu2gpuParams);
			cpu2gpuParams.waitForCreationToComplete();

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
		gpuMeshBuffer = std::get<0>(gpuRectangle);
		gpuRenderpassIndependentPipeline = std::get<1>(gpuRectangle);
		gpuubo = std::get<2>(gpuRectangle);
		gpuDescriptorSet1 = std::get<3>(gpuRectangle);
		gpuDescriptorSet3 = std::get<4>(gpuRectangle);
		gpuGraphicsPipeline = std::get<5>(gpuRectangle);

		logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, commandBuffers);

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
		{
			imageAcquire[i] = logicalDevice->createSemaphore();
			renderFinished[i] = logicalDevice->createSemaphore();
		}
	}

	/*
		Hot loop for rendering a scene.
	*/

	void workLoopBody() override
	{
		++resourceIx;
		if (resourceIx >= FRAMES_IN_FLIGHT)
			resourceIx = 0;

		auto& commandBuffer = commandBuffers[resourceIx];
		auto& fence = frameComplete[resourceIx];

		if (fence)
			while (logicalDevice->waitForFences(1u, &fence.get(), false, MAX_TIMEOUT) == video::IGPUFence::ES_TIMEOUT) {}
		else
			fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

		auto renderStart = std::chrono::system_clock::now();
		const auto renderDt = std::chrono::duration_cast<std::chrono::milliseconds>(renderStart - lastTime).count();
		lastTime = renderStart;
		{ // Calculate Simple Moving Average for FrameTime
			time_sum -= dtList[frame_count];
			time_sum += renderDt;
			dtList[frame_count] = renderDt;
			frame_count++;
			if (frame_count >= NBL_FRAMES_TO_AVERAGE)
			{
				frameDataFilled = true;
				frame_count = 0;
			}

		}
		const double averageFrameTime = frameDataFilled ? (time_sum / (double)NBL_FRAMES_TO_AVERAGE) : (time_sum / frame_count);

#ifdef NBL_MORE_LOGS
		logger->log("renderDt = %f ------ averageFrameTime = %f", system::ILogger::ELL_INFO, renderDt, averageFrameTime);
#endif // NBL_MORE_LOGS

		auto averageFrameTimeDuration = std::chrono::duration<double, std::milli>(averageFrameTime);
		auto nextPresentationTime = renderStart + averageFrameTimeDuration;
		auto nextPresentationTimeStamp = std::chrono::duration_cast<std::chrono::microseconds>(nextPresentationTime.time_since_epoch());

		inputSystem->getDefaultMouse(&mouse);
		inputSystem->getDefaultKeyboard(&keyboard);

		camera.beginInputProcessing(nextPresentationTimeStamp);
		mouse.consumeEvents([&](const ui::IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); }, logger.get());
		keyboard.consumeEvents([&](const ui::IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); }, logger.get());
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

		swapchain->acquireNextImage(MAX_TIMEOUT, imageAcquire[resourceIx].get(), nullptr, &acquiredNextFBO);

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
			beginInfo.framebuffer = fbo[acquiredNextFBO];
			beginInfo.renderpass = renderpass;
			beginInfo.renderArea = area;
			beginInfo.clearValues = clear;
		}

		commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);

		const auto viewProjection = camera.getConcatenatedMatrix();
		core::matrix3x4SIMD modelMatrix;
		modelMatrix.setRotation(nbl::core::quaternion(0, 1, 0));

		auto mv = core::concatenateBFollowedByA(camera.getViewMatrix(), modelMatrix);
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
		commandBuffer->updateBuffer(gpuubo.get(), 0ull, gpuubo->getSize(), &uboData);

		/*
			Binding the most important objects needed to
			render anything on the screen with textures:

			- gpu pipeline
			- gpu descriptor sets
		*/

		commandBuffer->bindGraphicsPipeline(gpuGraphicsPipeline.get());
		commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 1u, 1u, &gpuDescriptorSet1.get(), 0u);
		commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 3u, 1u, &gpuDescriptorSet3.get(), 0u);

		/*
			Drawing a mesh (created rectangle) with it's gpu mesh buffer usage.
		*/

		commandBuffer->drawMeshBuffer(gpuMeshBuffer.get());

		commandBuffer->endRenderPass();
		commandBuffer->end();

		CommonAPI::Submit(logicalDevice.get(), swapchain.get(), commandBuffer.get(), queues[CommonAPI::InitOutput::EQT_GRAPHICS], imageAcquire[resourceIx].get(), renderFinished[resourceIx].get(), fence.get());
		CommonAPI::Present(logicalDevice.get(), swapchain.get(), queues[CommonAPI::InitOutput::EQT_GRAPHICS], renderFinished[resourceIx].get(), acquiredNextFBO);
	}

	bool keepRunning() override
	{
		return windowCb->isWindowOpen();
	}
};

NBL_COMMON_API_MAIN(NablaTutorialExampleApp)
