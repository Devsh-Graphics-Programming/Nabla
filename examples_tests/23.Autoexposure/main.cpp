// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"

#include "nbl/ext/LumaMeter/CLumaMeter.h"
#include "nbl/ext/ToneMapper/CToneMapper.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"

using namespace nbl;

class AutoexposureSampleApp : public ApplicationBase
{
	constexpr static uint32_t WIN_W = 1280u;
	constexpr static uint32_t WIN_H = 720u;
	constexpr static uint32_t SC_IMG_COUNT = 3u;
	constexpr static uint32_t FRAMES_IN_FLIGHT = 5u;
	static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

	using LumaMeterClass = ext::LumaMeter::CLumaMeter;
	static constexpr LumaMeterClass::E_METERING_MODE MeterMode = LumaMeterClass::EMM_MEDIAN;
	static constexpr float MinLuma = 1.f / 2048.f;
	static constexpr float MaxLuma = 65536.f;

	using ToneMapperClass = ext::ToneMapper::CToneMapper;
	static constexpr ToneMapperClass::E_OPERATOR TMO = ToneMapperClass::EO_ACES;
	static constexpr float Exposure = 0.f;
	static constexpr float Key = 0.18;

	static constexpr bool usingLumaMeter = MeterMode < LumaMeterClass::EMM_COUNT;
	static constexpr bool usingTemporalAdapatation = true;

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
			fbos[i] = core::smart_refctd_ptr(f[i]);
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

	APP_CONSTRUCTOR(AutoexposureSampleApp);

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
		const video::ISurface::SFormat surfaceFormat(asset::EF_B8G8R8A8_SRGB, asset::ECP_SRGB, asset::EOTF_sRGB);

		CommonAPI::InitOutput initOutput;
		initOutput.window = core::smart_refctd_ptr(window);
		CommonAPI::Init(
			initOutput,
			video::EAT_VULKAN,
			"Autoexposure",
			requiredInstanceFeatures,
			optionalInstanceFeatures,
			requiredDeviceFeatures,
			optionalDeviceFeatures,
			WIN_W, WIN_H, SC_IMG_COUNT,
			swapchainImageUsage,
			surfaceFormat);

		system = std::move(initOutput.system);
		window = std::move(initOutput.window);
		windowCb = std::move(initOutput.windowCb);
		apiConnection = std::move(initOutput.apiConnection);
		surface = std::move(initOutput.surface);
		physicalDevice = std::move(initOutput.physicalDevice);
		logicalDevice = std::move(initOutput.logicalDevice);
		utilities = std::move(initOutput.utilities);
		queues = std::move(initOutput.queues);
		swapchain = std::move(initOutput.swapchain);
		assetManager = std::move(initOutput.assetManager);
		cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
		logger = std::move(initOutput.logger);
		inputSystem = std::move(initOutput.inputSystem);
		renderpass = std::move(initOutput.renderpass);
		fbos = std::move(initOutput.fbo);

		// find a queue family index which has both graphics and compute capability
		// (this can probably be bypassed by a queue-ownership transfer in the main rendering loop)
		uint32_t queueFamilyIndex = ~0u;
		{
			auto queueFamilyProps = logicalDevice->getPhysicalDevice()->getQueueFamilyProperties();
			for (uint32_t i = 0u; i < static_cast<uint32_t>(queueFamilyProps.size()); ++i)
			{
				const auto& prop = queueFamilyProps[i];
				if ((prop.queueFlags & video::IPhysicalDevice::EQF_COMPUTE_BIT).value && (prop.queueFlags & video::IPhysicalDevice::EQF_GRAPHICS_BIT).value)
					queueFamilyIndex = i;
			}

			if (queueFamilyIndex == ~0u)
			{
				logger->log("This example needs a queue family with BOTH graphics and compute capability to run!\n", system::ILogger::ELL_ERROR);
				exit(-1);
			}

			graphicsComputeQueue = logicalDevice->getQueue(queueFamilyIndex, 0u);
		}

		auto cmdpool = logicalDevice->createCommandPool(queueFamilyIndex, video::IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
		logicalDevice->createCommandBuffers(cmdpool.get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, cmdbufs);

		const char* imagePath = "../../media/noises/spp_benchmark_4k_512.exr";
		{
			asset::IAssetLoader::SAssetLoadParams params = {};
			logger->log("Loading image at path: %s\n", system::ILogger::ELL_DEBUG, imagePath);
			asset::SAssetBundle imageBundle = assetManager->getAsset(imagePath, params);
			if (!imageBundle.getContents().begin())
			{
				logger->log("Failed to load image at path %s!\n", system::ILogger::ELL_ERROR, imagePath);
				exit(-1);
			}
			logger->log("Image loaded successfully.\n");
		
			auto cpuImg = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(imageBundle.getContents().begin()[0]);

			cpuImg->addImageUsageFlags(asset::IImage::EUF_SAMPLED_BIT);
			asset::ICPUImage::SCreationParams imgInfo = cpuImg->getCreationParameters();

			cpu2gpuParams.beginCommandBuffers();
			auto gpuArray = cpu2gpu.getGPUObjectsFromAssets(&cpuImg.get(), &cpuImg.get() + 1, cpu2gpuParams);
			if (!gpuArray || gpuArray->size() < 1u || !(*gpuArray)[0])
			{
				logger->log("Failed to convert CPU image to GPU image!\n", system::ILogger::ELL_ERROR);
				exit(-1);
			}
			cpu2gpuParams.waitForCreationToComplete();

			auto gpuImage = gpuArray->operator[](0u);

			video::IGPUImageView::SCreationParams imgViewInfo;
			imgViewInfo.flags = static_cast<video::IGPUImageView::E_CREATE_FLAGS>(0u);
			imgViewInfo.image = std::move(gpuImage);
			imgViewInfo.viewType = video::IGPUImageView::ET_2D_ARRAY;
			imgViewInfo.format = imgViewInfo.image->getCreationParameters().format;
			imgViewInfo.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			imgViewInfo.subresourceRange.baseMipLevel = 0;
			imgViewInfo.subresourceRange.levelCount = 1;
			imgViewInfo.subresourceRange.baseArrayLayer = 0;
			imgViewInfo.subresourceRange.layerCount = 1;
			imageToTonemapView = logicalDevice->createGPUImageView(std::move(imgViewInfo));
		}

		const auto inputImageFormat = imageToTonemapView->getCreationParameters().image->getCreationParameters().format;
		const auto outputImageFormat = swapchain->getCreationParameters().surfaceFormat.format;
		const auto inputColorSpace = std::make_tuple(inputImageFormat, asset::ECP_SRGB, asset::EOTF_IDENTITY);
		const auto outputColorSpace = std::make_tuple(outputImageFormat, asset::ECP_SRGB, asset::OETF_sRGB);

		// create tonemapping compute pipeline
		{
			auto cpuSpecShader = ToneMapperClass::createShader(
				assetManager->getGLSLCompiler(),
				inputColorSpace,
				outputColorSpace,
				TMO,
				usingLumaMeter,
				MeterMode,
				MinLuma, MaxLuma,
				usingTemporalAdapatation);

			auto gpuUnspecShader = logicalDevice->createGPUShader(core::smart_refctd_ptr<asset::ICPUShader>(cpuSpecShader->getUnspecialized()));
			auto gpuSpecShader = logicalDevice->createGPUSpecializedShader(gpuUnspecShader.get(), cpuSpecShader->getSpecializationInfo());
		
			auto pipelineLayout = ToneMapperClass::getDefaultPipelineLayout(logicalDevice.get(), usingLumaMeter);
			tonemappingPipeline = logicalDevice->createGPUComputePipeline(nullptr, std::move(pipelineLayout), std::move(gpuSpecShader));
		}

		const auto outputImageViewFormat = ToneMapperClass::getOutputViewFormat(outputImageFormat);
		if (outputImageViewFormat == asset::EF_UNKNOWN)
		{
			logger->log("Input image format not supported!\n", system::ILogger::ELL_ERROR);
			exit(-1);
		}

		// create tonemapper output image views
		for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
		{
			auto creationParams = imageToTonemapView->getCreationParameters().image->getCreationParameters();
			creationParams.format = outputImageViewFormat;
			creationParams.usage = static_cast<video::IGPUImage::E_USAGE_FLAGS>(video::IGPUImage::EUF_STORAGE_BIT | video::IGPUImage::EUF_SAMPLED_BIT);
			auto outImage = logicalDevice->createDeviceLocalGPUImageOnDedMem(std::move(creationParams));

			// transition the image to GENERAL
			{
				const auto& imageCreationParams = outImage->getCreationParameters();

				const auto& cb = cmdbufs[0];

				video::IGPUCommandBuffer::SImageMemoryBarrier toGeneral = {};
				toGeneral.image = outImage;
				toGeneral.oldLayout = asset::EIL_UNDEFINED;
				toGeneral.newLayout = asset::EIL_GENERAL;
				toGeneral.subresourceRange.aspectMask = video::IGPUImage::EAF_COLOR_BIT;
				toGeneral.subresourceRange.baseArrayLayer = 0u;
				toGeneral.subresourceRange.layerCount = imageCreationParams.arrayLayers;
				toGeneral.subresourceRange.baseMipLevel = 0u;
				toGeneral.subresourceRange.levelCount = imageCreationParams.mipLevels;

				cb->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
				cb->pipelineBarrier(
					asset::EPSF_TOP_OF_PIPE_BIT,
					asset::EPSF_BOTTOM_OF_PIPE_BIT,
					asset::EDF_BY_REGION_BIT,
					0u, nullptr,
					0u, nullptr,
					1u, &toGeneral);
				cb->end();

				auto fence = logicalDevice->createFence(video::IGPUFence::ECF_UNSIGNALED);

				video::IGPUQueue::SSubmitInfo submit;
				submit.commandBufferCount = 1u;
				submit.commandBuffers = &cb.get();
				graphicsComputeQueue->submit(1u, &submit, fence.get());

				logicalDevice->blockForFences(1u, &fence.get());
			}

			auto viewCreationParams = imageToTonemapView->getCreationParameters();
			viewCreationParams.format = outImage->getCreationParameters().format;
			viewCreationParams.image = outImage;
			tonemapperOutImageViews[i] = logicalDevice->createGPUImageView(std::move(viewCreationParams));
		}

		const float meteringMinUV[2] = { 0.1f,0.1f };
		const float meteringMaxUV[2] = { 0.9f,0.9f };
		LumaMeterClass::Uniforms_t<MeterMode> uniforms;
		lumaDispatchInfo = LumaMeterClass::buildParameters(
			uniforms,
			imageToTonemapView->getCreationParameters().image->getCreationParameters().extent,
			meteringMinUV, meteringMaxUV);


		// create luma meter ubo
		{
			const size_t bufferSize = sizeof(uniforms);
			video::IGPUBuffer::SCreationParams creationParams = {};
			creationParams.usage = static_cast<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::EUF_UNIFORM_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT);
			lumaMeterUbo = logicalDevice->createDeviceLocalGPUBufferOnDedMem(creationParams, bufferSize);

			// fill it up with `uniforms`
			core::smart_refctd_ptr<video::IGPUFence> fence = logicalDevice->createFence(video::IGPUFence::ECF_UNSIGNALED);
			asset::SBufferRange<video::IGPUBuffer> bufferRange;
			bufferRange.offset = 0ull;
			bufferRange.size = lumaMeterUbo->getCachedCreationParams().declaredSize;
			bufferRange.buffer = lumaMeterUbo;
			utilities->updateBufferRangeViaStagingBuffer(
				fence.get(),
				queues[CommonAPI::InitOutput::EQT_TRANSFER_UP],
				bufferRange,
				&uniforms);
			logicalDevice->blockForFences(1u, &fence.get());
		}

		// create tone params luma output buffers
		{
			tonemappingParams.setAdaptationFactorFromFrameDelta(0.f);

			const size_t parameterBufferSize = ToneMapperClass::getParameterBufferSize<TMO, MeterMode>();

			video::IGPUBuffer::SCreationParams creationParams = {};
			creationParams.usage = static_cast<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT);

			for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
			{
				toneParamsLumaOutputBuffer[i] = logicalDevice->createDeviceLocalGPUBufferOnDedMem(creationParams, parameterBufferSize);

				core::smart_refctd_ptr<video::IGPUFence> fence = logicalDevice->createFence(video::IGPUFence::ECF_UNSIGNALED);
				asset::SBufferRange<video::IGPUBuffer> bufferRange;
				bufferRange.offset = 0ull;
				bufferRange.size = sizeof(tonemappingParams);
				bufferRange.buffer = toneParamsLumaOutputBuffer[i];
				utilities->updateBufferRangeViaStagingBuffer(
					fence.get(),
					queues[CommonAPI::InitOutput::EQT_TRANSFER_UP],
					bufferRange,
					&tonemappingParams);
				logicalDevice->blockForFences(1u, &fence.get());
			}
		}

		// create and update tonemapping descriptor sets
		{
			const auto pipelineLayout = tonemappingPipeline->getLayout();
			const auto dsLayout = pipelineLayout->getDescriptorSetLayout(0u);

			const uint32_t setCount = 3u;
			auto dsPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &dsLayout, &dsLayout + 1ull, &setCount);

			for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
			{
				tonemappingDS[i] = logicalDevice->createGPUDescriptorSet(dsPool.get(), core::smart_refctd_ptr<const video::IGPUDescriptorSetLayout>(dsLayout));

				ToneMapperClass::updateDescriptorSet<MeterMode>(
					logicalDevice.get(),
					tonemappingDS[i].get(),
					tonemapperOutImageViews[i],
					toneParamsLumaOutputBuffer[i],
					imageToTonemapView,
					lumaMeterUbo);
			}
		}

		// create luma metering pipeline
		{
			auto cpuSpecShader = LumaMeterClass::createShader(assetManager->getGLSLCompiler(), inputColorSpace, MeterMode, MinLuma, MaxLuma);
			auto gpuUnspecShader = logicalDevice->createGPUShader(core::smart_refctd_ptr<asset::ICPUShader>(cpuSpecShader->getUnspecialized()));
			auto gpuSpecShader = logicalDevice->createGPUSpecializedShader(gpuUnspecShader.get(), cpuSpecShader->getSpecializationInfo());

			auto pipelineLayout = LumaMeterClass::getDefaultPipelineLayout(logicalDevice.get());
			lumaMeteringPipeline = logicalDevice->createGPUComputePipeline(nullptr, std::move(pipelineLayout), std::move(gpuSpecShader));
		}

		// create and update luma metering descriptor sets
		{
			auto dsLayout = core::smart_refctd_ptr<const video::IGPUDescriptorSetLayout>(lumaMeteringPipeline->getLayout()->getDescriptorSetLayout(0u));
			const uint32_t setCount = SC_IMG_COUNT;
			auto descriptorPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &dsLayout.get(), &dsLayout.get() + 1ull, &setCount);

			for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
			{
				lumaMeteringDS[i] = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), core::smart_refctd_ptr(dsLayout));

				asset::SBufferRange<video::IGPUBuffer> lumaOutputBufferRange = {};
				lumaOutputBufferRange.offset = sizeof(ToneMapperClass::Params_t<TMO>);
				lumaOutputBufferRange.size = toneParamsLumaOutputBuffer[i]->getCachedCreationParams().declaredSize - lumaOutputBufferRange.offset;
				lumaOutputBufferRange.buffer = toneParamsLumaOutputBuffer[i];

				LumaMeterClass::updateDescriptorSet<MeterMode>(
					lumaMeteringDS[i].get(),
					lumaMeterUbo,
					lumaOutputBufferRange,
					imageToTonemapView,
					logicalDevice.get());
			}
		}

		// create FST fragment shader
		const char* fstFragmentShaderPath = "../fstFrag.frag";
		core::smart_refctd_ptr<video::IGPUSpecializedShader> fstFragShaderGPU = nullptr;
		{
			auto fs_bundle = assetManager->getAsset(fstFragmentShaderPath, {});
			auto fs_contents = fs_bundle.getContents();
			if (fs_contents.empty())
				assert(false);

			asset::ICPUSpecializedShader* cpuFragmentShader = static_cast<asset::ICPUSpecializedShader*>(fs_contents.begin()->get());

			{
				cpu2gpuParams.beginCommandBuffers();
				auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&cpuFragmentShader, &cpuFragmentShader + 1, cpu2gpuParams);
				cpu2gpuParams.waitForCreationToComplete(false);

				if (!gpu_array.get() || gpu_array->size() < 1u || !(*gpu_array)[0])
					assert(false);

				fstFragShaderGPU = (*gpu_array)[0];
			}
		}

		// create FST DS and pipeline layout
		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> fstDSLayout = nullptr;
		core::smart_refctd_ptr<video::IGPUPipelineLayout> fstPipelineLayout = nullptr;
		{
			asset::ISampler::SParams samplerParams =
			{
				asset::ISampler::ETC_CLAMP_TO_EDGE,
				asset::ISampler::ETC_CLAMP_TO_EDGE,
				asset::ISampler::ETC_CLAMP_TO_EDGE,
				asset::ISampler::ETBC_FLOAT_OPAQUE_BLACK,
				asset::ISampler::ETF_NEAREST,
				asset::ISampler::ETF_NEAREST,
				asset::ISampler::ESMM_LINEAR,
				0u,
				false,
				nbl::asset::ECO_ALWAYS
			};
			core::smart_refctd_ptr<video::IGPUSampler> sampler = logicalDevice->createGPUSampler(std::move(samplerParams));
			video::IGPUDescriptorSetLayout::SBinding binding =
			{
				0u,
				nbl::asset::EDT_COMBINED_IMAGE_SAMPLER,
				1u,
				nbl::video::IGPUShader::ESS_FRAGMENT,
				&sampler
			};

			fstDSLayout = logicalDevice->createGPUDescriptorSetLayout(&binding, &binding + 1u);
			fstPipelineLayout = logicalDevice->createGPUPipelineLayout(nullptr, nullptr, nullptr, nullptr, nullptr, core::smart_refctd_ptr(fstDSLayout));
		}

		// create FST renderpass independent pipeline
		auto fstProtoPipeline = nbl::ext::FullScreenTriangle::createProtoPipeline(cpu2gpuParams);
		auto fstRenderpassIndep = ext::FullScreenTriangle::createRenderpassIndependentPipeline(
			logicalDevice.get(),
			fstProtoPipeline,
			std::move(fstFragShaderGPU),
			std::move(fstPipelineLayout));

		// create & update FST DS
		{
			const uint32_t setCount = SC_IMG_COUNT;
			auto dsPool = logicalDevice->createDescriptorPoolForDSLayouts(
				video::IDescriptorPool::ECF_NONE,
				&fstDSLayout.get(), &fstDSLayout.get() + 1ull,
				&setCount);

			for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
			{
				fstDS[i] = logicalDevice->createGPUDescriptorSet(dsPool.get(), core::smart_refctd_ptr(fstDSLayout));

				video::IGPUDescriptorSet::SWriteDescriptorSet write = {};
				write.dstSet = fstDS[i].get();
				write.binding = 0u;
				write.arrayElement = 0u;
				write.count = 1u;
				write.descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;

				video::IGPUDescriptorSet::SDescriptorInfo info = {};
				info.desc = tonemapperOutImageViews[i];
				info.image.sampler = nullptr;
				info.image.imageLayout = asset::EIL_SHADER_READ_ONLY_OPTIMAL;

				write.info = &info;

				logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);

			}
		}

		// create FST graphics pipline
		{
			nbl::video::IGPUGraphicsPipeline::SCreationParams creationParams = {};
			creationParams.renderpassIndependent = core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(const_cast<video::IGPURenderpassIndependentPipeline*>(fstRenderpassIndep.get()));
			creationParams.renderpass = core::smart_refctd_ptr(renderpass);

			fstGraphicsPipeline = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(creationParams));
		}

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
		{
			imageAcquire[i] = logicalDevice->createSemaphore();
			renderFinished[i] = logicalDevice->createSemaphore();
		}

		lastPresentStamp = std::chrono::high_resolution_clock::now();
	}

	void onAppTerminated_impl() override
	{
		logicalDevice->waitIdle();
	}

	void workLoopBody() override
	{
		++resourceIx;
		if (resourceIx >= FRAMES_IN_FLIGHT)
			resourceIx = 0;

		auto& cb = cmdbufs[resourceIx];
		auto& fence = frameComplete[resourceIx];

		if (fence)
		{
			logicalDevice->blockForFences(1u, &fence.get(), true);
			logicalDevice->resetFences(1u, &fence.get());
		}
		else
			fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

		cb->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
		asset::SViewport viewport;
		viewport.minDepth = 1.f;
		viewport.maxDepth = 0.f;
		viewport.x = 0u;
		viewport.y = 0u;
		viewport.width = WIN_W;
		viewport.height = WIN_H;
		cb->setViewport(0u, 1u, &viewport);

		VkRect2D scissor;
		scissor.offset = { 0, 0 };
		scissor.extent = { WIN_W, WIN_H };
		cb->setScissor(0u, 1u, &scissor);

		swapchain->acquireNextImage(MAX_TIMEOUT, imageAcquire[resourceIx].get(), nullptr, &acquiredNextFBO);
		cb->pushConstants(lumaMeteringPipeline->getLayout(), asset::IShader::ESS_COMPUTE, 0u, sizeof(outBufferIx[acquiredNextFBO]), &outBufferIx[acquiredNextFBO]);
		outBufferIx[acquiredNextFBO] ^= 0x1u;
		cb->bindDescriptorSets(asset::EPBP_COMPUTE, lumaMeteringPipeline->getLayout(), 0u, 1u, &lumaMeteringDS[acquiredNextFBO].get());
		cb->bindComputePipeline(lumaMeteringPipeline.get());

		video::IGPUCommandBuffer::SBufferMemoryBarrier lumaOutputBufferBarrier = {};
		{
			lumaOutputBufferBarrier.barrier.srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0u);
			lumaOutputBufferBarrier.barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
			lumaOutputBufferBarrier.srcQueueFamilyIndex = ~0u;
			lumaOutputBufferBarrier.dstQueueFamilyIndex = ~0u;
			lumaOutputBufferBarrier.buffer = toneParamsLumaOutputBuffer[acquiredNextFBO];
			lumaOutputBufferBarrier.offset = sizeof(ToneMapperClass::Params_t<TMO>);
			lumaOutputBufferBarrier.size = toneParamsLumaOutputBuffer[acquiredNextFBO]->getCachedCreationParams().declaredSize - lumaOutputBufferBarrier.offset;
		}

		LumaMeterClass::dispatchHelper(
			cb.get(),
			lumaDispatchInfo,
			asset::EPSF_TOP_OF_PIPE_BIT,
			0u, nullptr,
			asset::EPSF_COMPUTE_SHADER_BIT,
			1u, &lumaOutputBufferBarrier);

		cb->bindDescriptorSets(asset::EPBP_COMPUTE, tonemappingPipeline->getLayout(), 0u, 1u, &tonemappingDS[acquiredNextFBO].get());
		cb->bindComputePipeline(tonemappingPipeline.get());

		video::IGPUCommandBuffer::SImageMemoryBarrier layoutTransBarrier = {};
		{
			const auto& imageCreationParams = tonemapperOutImageViews[acquiredNextFBO]->getCreationParameters().image->getCreationParameters();

			layoutTransBarrier.barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
			layoutTransBarrier.barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
			layoutTransBarrier.oldLayout = asset::EIL_GENERAL;
			layoutTransBarrier.newLayout = asset::EIL_SHADER_READ_ONLY_OPTIMAL;
			layoutTransBarrier.srcQueueFamilyIndex = ~0u;
			layoutTransBarrier.dstQueueFamilyIndex = ~0u;
			layoutTransBarrier.image = tonemapperOutImageViews[acquiredNextFBO]->getCreationParameters().image;
			layoutTransBarrier.subresourceRange.aspectMask = video::IGPUImage::EAF_COLOR_BIT;
			layoutTransBarrier.subresourceRange.baseArrayLayer = 0u;
			layoutTransBarrier.subresourceRange.layerCount = imageCreationParams.arrayLayers;
			layoutTransBarrier.subresourceRange.baseMipLevel = 0u;
			layoutTransBarrier.subresourceRange.levelCount = imageCreationParams.mipLevels;
		}

		ToneMapperClass::dispatchHelper(
			cb.get(),
			tonemapperOutImageViews[acquiredNextFBO].get(),
			asset::EPSF_TOP_OF_PIPE_BIT,
			0u, nullptr,
			asset::EPSF_FRAGMENT_SHADER_BIT,
			1u, &layoutTransBarrier);

		// This can probably be sucked into ToneMapperClass::dispatchHelper
		if (usingTemporalAdapatation)
		{
			auto thisPresentStamp = std::chrono::high_resolution_clock::now();
			auto microsecondsElapsedBetweenPresents = std::chrono::duration_cast<std::chrono::microseconds>(thisPresentStamp - lastPresentStamp);
			lastPresentStamp = thisPresentStamp;

			tonemappingParams.setAdaptationFactorFromFrameDelta(float(microsecondsElapsedBetweenPresents.count()) / 1000000.f);

			// dont override shader output
			constexpr auto offsetPastLumaHistory = offsetof(decltype(tonemappingParams), lastFrameExtraEVAsHalf) + sizeof(decltype(tonemappingParams)::lastFrameExtraEVAsHalf);
			auto* paramPtr = reinterpret_cast<const uint8_t*>(&tonemappingParams);
			asset::SBufferRange<video::IGPUBuffer> bufferRange = {};
			bufferRange.buffer = toneParamsLumaOutputBuffer[acquiredNextFBO];
			bufferRange.offset = offsetPastLumaHistory;
			bufferRange.size = sizeof(tonemappingParams) - bufferRange.offset;

			// memory dependency to ensure all the compute shader read for the tonemapping dispatch
			// has finished before we try to transfer new stuff to the buffer
			video::IGPUCommandBuffer::SBufferMemoryBarrier prevParamsReadBarrier = {};
			prevParamsReadBarrier.barrier.srcAccessMask = asset::EAF_SHADER_READ_BIT;
			prevParamsReadBarrier.barrier.dstAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
			prevParamsReadBarrier.srcQueueFamilyIndex = ~0u;
			prevParamsReadBarrier.dstQueueFamilyIndex = ~0u;
			prevParamsReadBarrier.buffer = bufferRange.buffer;
			prevParamsReadBarrier.offset = bufferRange.offset;
			prevParamsReadBarrier.size = bufferRange.size;

			cb->pipelineBarrier(
				asset::EPSF_COMPUTE_SHADER_BIT,
				asset::EPSF_TRANSFER_BIT,
				asset::EDF_BY_REGION_BIT,
				0u, nullptr,
				1u, &prevParamsReadBarrier,
				0u, nullptr);

			uint32_t waitSemCount = 0u;
			video::IGPUSemaphore *const* waitSem = nullptr;
			const asset::E_PIPELINE_STAGE_FLAGS* waitStages = nullptr;
			utilities->updateBufferRangeViaStagingBuffer(
				cb.get(),
				nullptr,
				graphicsComputeQueue,
				bufferRange,
				paramPtr + offsetPastLumaHistory,
				waitSemCount, waitSem, waitStages);
		}

		video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo = {};
		{
			VkRect2D area;
			area.offset = { 0,0 };
			area.extent = { WIN_W, WIN_H };
			asset::SClearValue clear[2] = {};
			clear[0].color.float32[0] = 1.f;
			clear[0].color.float32[1] = 0.f;
			clear[0].color.float32[2] = 0.f;
			clear[0].color.float32[3] = 1.f;
			clear[1].depthStencil.depth = 0.f;

			beginInfo.clearValueCount = 2u;
			beginInfo.framebuffer = fbos[acquiredNextFBO];
			beginInfo.renderpass = renderpass;
			beginInfo.renderArea = area;
			beginInfo.clearValues = clear;
		}

		cb->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);
		cb->bindDescriptorSets(
			asset::EPBP_GRAPHICS,
			fstGraphicsPipeline->getRenderpassIndependentPipeline()->getLayout(),
			3, 1,
			&fstDS[acquiredNextFBO].get());
		cb->bindGraphicsPipeline(fstGraphicsPipeline.get());

		ext::FullScreenTriangle::recordDrawCalls(cb.get());

		cb->endRenderPass();

		// transfer back to GENERAL
		{
			layoutTransBarrier.barrier.srcAccessMask = asset::EAF_SHADER_READ_BIT;
			layoutTransBarrier.barrier.dstAccessMask = asset::EAF_SHADER_WRITE_BIT;
			layoutTransBarrier.oldLayout = asset::EIL_SHADER_READ_ONLY_OPTIMAL;
			layoutTransBarrier.newLayout = asset::EIL_GENERAL;
		}
		cb->pipelineBarrier(
			asset::EPSF_FRAGMENT_SHADER_BIT,
			asset::EPSF_COMPUTE_SHADER_BIT,
			asset::EDF_BY_REGION_BIT,
			0u, nullptr,
			0u, nullptr,
			1u, &layoutTransBarrier);

		cb->end();

		CommonAPI::Submit(
			logicalDevice.get(),
			swapchain.get(),
			cb.get(),
			graphicsComputeQueue,
			imageAcquire[resourceIx].get(),
			renderFinished[resourceIx].get(),
			fence.get());

		CommonAPI::Present(
			logicalDevice.get(),
			swapchain.get(),
			graphicsComputeQueue,
			renderFinished[resourceIx].get(),
			acquiredNextFBO);
	}

	bool keepRunning() override
	{
		return windowCb->isWindowOpen();
	}

private:
	core::smart_refctd_ptr<nbl::ui::IWindow> window;
	core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
	core::smart_refctd_ptr<nbl::video::IAPIConnection> apiConnection;
	core::smart_refctd_ptr<nbl::video::ISurface> surface;
	core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
	core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
	video::IPhysicalDevice* physicalDevice;
	std::array<video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> queues;
	video::IGPUQueue* graphicsComputeQueue;
	core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
	core::smart_refctd_ptr<video::IGPURenderpass> renderpass = nullptr;
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, CommonAPI::InitOutput::MaxSwapChainImageCount> fbos;
	core::smart_refctd_ptr<nbl::system::ISystem> system;
	core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
	video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
	core::smart_refctd_ptr<nbl::system::ILogger> logger;
	core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;
	video::IGPUObjectFromAssetConverter cpu2gpu;

	int32_t resourceIx = -1;
	uint32_t acquiredNextFBO = {};

	uint32_t outBufferIx[SC_IMG_COUNT] = { 0u, 0u, 0u };
	std::chrono::high_resolution_clock::time_point lastPresentStamp;

	core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbufs[FRAMES_IN_FLIGHT] = { nullptr };

	core::smart_refctd_ptr<video::IGPUImageView> imageToTonemapView = nullptr;
	core::smart_refctd_ptr<video::IGPUImageView> quantStorageImageViews[SC_IMG_COUNT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUImageView> tonemapperOutImageViews[SC_IMG_COUNT] = {};
	core::smart_refctd_ptr<video::IGPUBuffer> lumaMeterUbo = nullptr;
	core::smart_refctd_ptr<video::IGPUBuffer> toneParamsLumaOutputBuffer[SC_IMG_COUNT] = {};

	ToneMapperClass::Params_t<TMO> tonemappingParams = ToneMapperClass::Params_t<TMO>(Exposure, Key, 0.85f);

	core::smart_refctd_ptr<video::IGPUDescriptorSet> lumaMeteringDS[SC_IMG_COUNT] = { };
	core::smart_refctd_ptr<video::IGPUDescriptorSet> tonemappingDS[SC_IMG_COUNT] = { };
	core::smart_refctd_ptr<video::IGPUComputePipeline> lumaMeteringPipeline = nullptr;
	core::smart_refctd_ptr<video::IGPUComputePipeline> tonemappingPipeline = nullptr;
	LumaMeterClass::DispatchInfo_t lumaDispatchInfo;

	core::smart_refctd_ptr<video::IGPUDescriptorSet> fstDS[SC_IMG_COUNT] = {};
	core::smart_refctd_ptr<nbl::video::IGPUGraphicsPipeline> fstGraphicsPipeline = nullptr;
};

NBL_COMMON_API_MAIN(AutoexposureSampleApp)

extern "C" {  _declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; }

#if 0
int main()
{
	nbl::SIrrlichtCreationParameters deviceParams;
	deviceParams.Bits = 24; //may have to set to 32bit for some platforms
	deviceParams.ZBufferBits = 24; //we'd like 32bit here
	deviceParams.DriverType = EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	deviceParams.WindowSize = dimension2d<uint32_t>(1280, 720);
	deviceParams.Fullscreen = false;
	deviceParams.Vsync = true; //! If supported by target platform
	deviceParams.Doublebuffer = true;
	deviceParams.Stencilbuffer = false; //! This will not even be a choice soon

	auto device = createDeviceEx(deviceParams);
	if (!device)
		return 1; // could not create selected driver.

	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);

	IVideoDriver* driver = device->getVideoDriver();
	
	nbl::io::IFileSystem* filesystem = device->getFileSystem();
	IAssetManager* am = device->getAssetManager();

	IAssetLoader::SAssetLoadParams lp;
	auto imageBundle = am->getAsset("../../media/noises/spp_benchmark_4k_512.exr", lp);

	E_FORMAT inFormat;
	constexpr auto outFormat = EF_R8G8B8A8_SRGB;
	smart_refctd_ptr<IGPUImage> outImg;
	smart_refctd_ptr<IGPUImageView> imgToTonemapView,outImgView;
	{
		auto cpuImg = IAsset::castDown<ICPUImage>(imageBundle.getContents().begin()[0]);
		IGPUImage::SCreationParams imgInfo = cpuImg->getCreationParameters();
		inFormat = imgInfo.format;

		auto gpuImages = driver->getGPUObjectsFromAssets(&cpuImg.get(),&cpuImg.get()+1);
		auto gpuImage = gpuImages->operator[](0u);

		IGPUImageView::SCreationParams imgViewInfo;
		imgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		imgViewInfo.image = std::move(gpuImage);
		imgViewInfo.viewType = IGPUImageView::ET_2D_ARRAY;
		imgViewInfo.format = inFormat;
		imgViewInfo.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0u);
		imgViewInfo.subresourceRange.baseMipLevel = 0;
		imgViewInfo.subresourceRange.levelCount = 1;
		imgViewInfo.subresourceRange.baseArrayLayer = 0;
		imgViewInfo.subresourceRange.layerCount = 1;
		imgToTonemapView = driver->createGPUImageView(IGPUImageView::SCreationParams(imgViewInfo));

		imgInfo.format = outFormat;
		outImg = driver->createDeviceLocalGPUImageOnDedMem(std::move(imgInfo));

		imgViewInfo.image = outImg;
		imgViewInfo.format = outFormat;
		outImgView = driver->createGPUImageView(IGPUImageView::SCreationParams(imgViewInfo));
	}

	auto glslCompiler = am->getGLSLCompiler();
	const auto inputColorSpace = std::make_tuple(inFormat,ECP_SRGB,EOTF_IDENTITY);

	using LumaMeterClass = ext::LumaMeter::CLumaMeter;
	constexpr auto MeterMode = LumaMeterClass::EMM_MEDIAN;
	const float minLuma = 1.f/2048.f;
	const float maxLuma = 65536.f;

	auto cpuLumaMeasureSpecializedShader = LumaMeterClass::createShader(glslCompiler,inputColorSpace,MeterMode,minLuma,maxLuma);
	auto gpuLumaMeasureShader = driver->createGPUShader(smart_refctd_ptr<const ICPUShader>(cpuLumaMeasureSpecializedShader->getUnspecialized()));
	auto gpuLumaMeasureSpecializedShader = driver->createGPUSpecializedShader(gpuLumaMeasureShader.get(), cpuLumaMeasureSpecializedShader->getSpecializationInfo());

	const float meteringMinUV[2] = { 0.1f,0.1f };
	const float meteringMaxUV[2] = { 0.9f,0.9f };
	LumaMeterClass::Uniforms_t<MeterMode> uniforms;
	auto lumaDispatchInfo = LumaMeterClass::buildParameters(uniforms, outImg->getCreationParameters().extent, meteringMinUV, meteringMaxUV);

	auto uniformBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(uniforms),&uniforms);


	using ToneMapperClass = ext::ToneMapper::CToneMapper;
	constexpr auto TMO = ToneMapperClass::EO_ACES;
	constexpr bool usingLumaMeter = MeterMode<LumaMeterClass::EMM_COUNT;
	constexpr bool usingTemporalAdapatation = true;

	auto cpuTonemappingSpecializedShader = ToneMapperClass::createShader(am->getGLSLCompiler(),
		inputColorSpace,
		std::make_tuple(outFormat,ECP_SRGB,OETF_sRGB),
		TMO,usingLumaMeter,MeterMode,minLuma,maxLuma,usingTemporalAdapatation
	);
	auto gpuTonemappingShader = driver->createGPUShader(smart_refctd_ptr<const ICPUShader>(cpuTonemappingSpecializedShader->getUnspecialized()));
	auto gpuTonemappingSpecializedShader = driver->createGPUSpecializedShader(gpuTonemappingShader.get(),cpuTonemappingSpecializedShader->getSpecializationInfo());

	auto outImgStorage = ToneMapperClass::createViewForImage(driver,false,core::smart_refctd_ptr(outImg),{static_cast<IImage::E_ASPECT_FLAGS>(0u),0,1,0,1});

	auto parameterBuffer = driver->createDeviceLocalGPUBufferOnDedMem(ToneMapperClass::getParameterBufferSize<TMO,MeterMode>());
	constexpr float Exposure = 0.f;
	constexpr float Key = 0.18;
	auto params = ToneMapperClass::Params_t<TMO>(Exposure, Key, 0.85f);
	{
		params.setAdaptationFactorFromFrameDelta(0.f);
		driver->updateBufferRangeViaStagingBuffer(parameterBuffer.get(),0u,sizeof(params),&params);
	}

	auto commonPipelineLayout = ToneMapperClass::getDefaultPipelineLayout(driver,usingLumaMeter);

	auto lumaMeteringPipeline = driver->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(commonPipelineLayout),std::move(gpuLumaMeasureSpecializedShader));
	auto toneMappingPipeline = driver->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(commonPipelineLayout),std::move(gpuTonemappingSpecializedShader));

	auto commonDescriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(commonPipelineLayout->getDescriptorSetLayout(0u)));
	ToneMapperClass::updateDescriptorSet<TMO,MeterMode>(driver,commonDescriptorSet.get(),parameterBuffer,imgToTonemapView,outImgStorage,1u,2u,usingLumaMeter ? 3u:0u,uniformBuffer,0u,usingTemporalAdapatation);


	constexpr auto dynOffsetArrayLen = usingLumaMeter ? 2u : 1u;

	auto lumaDynamicOffsetArray = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<uint32_t> >(dynOffsetArrayLen,0u);
	lumaDynamicOffsetArray->back() = sizeof(ToneMapperClass::Params_t<TMO>);

	auto toneDynamicOffsetArray = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<uint32_t> >(dynOffsetArrayLen,0u);


	auto blitFBO = driver->addFrameBuffer();
	blitFBO->attach(video::EFAP_COLOR_ATTACHMENT0, std::move(outImgView));

	uint32_t outBufferIx = 0u;
	auto lastPresentStamp = std::chrono::high_resolution_clock::now();
	while (device->run() && receiver.keepOpen())
	{
		driver->beginScene(false, false);

		driver->bindComputePipeline(lumaMeteringPipeline.get());
		driver->bindDescriptorSets(EPBP_COMPUTE,commonPipelineLayout.get(),0u,1u,&commonDescriptorSet.get(),&lumaDynamicOffsetArray);
		driver->pushConstants(commonPipelineLayout.get(),IGPUSpecializedShader::ESS_COMPUTE,0u,sizeof(outBufferIx),&outBufferIx); outBufferIx ^= 0x1u;
		LumaMeterClass::dispatchHelper(driver,lumaDispatchInfo,true);

		driver->bindComputePipeline(toneMappingPipeline.get());
		driver->bindDescriptorSets(EPBP_COMPUTE,commonPipelineLayout.get(),0u,1u,&commonDescriptorSet.get(),&toneDynamicOffsetArray);
		ToneMapperClass::dispatchHelper(driver,outImgStorage.get(),true);

		driver->blitRenderTargets(blitFBO, nullptr, false, false);

		driver->endScene();
		if (usingTemporalAdapatation)
		{
			auto thisPresentStamp = std::chrono::high_resolution_clock::now();
			auto microsecondsElapsedBetweenPresents = std::chrono::duration_cast<std::chrono::microseconds>(thisPresentStamp-lastPresentStamp);
			lastPresentStamp = thisPresentStamp;

			params.setAdaptationFactorFromFrameDelta(float(microsecondsElapsedBetweenPresents.count())/1000000.f);
			// dont override shader output
			constexpr auto offsetPastLumaHistory = offsetof(decltype(params),lastFrameExtraEVAsHalf)+sizeof(decltype(params)::lastFrameExtraEVAsHalf);
			auto* paramPtr = reinterpret_cast<const uint8_t*>(&params);
			driver->updateBufferRangeViaStagingBuffer(parameterBuffer.get(), offsetPastLumaHistory, sizeof(params)-offsetPastLumaHistory, paramPtr+offsetPastLumaHistory);
		}
	}

	return 0;
}
#endif