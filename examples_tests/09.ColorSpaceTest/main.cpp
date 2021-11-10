// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>

#include "../common/CommonAPI.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"

using namespace nbl;

#define SWITCH_IMAGES_PER_X_MILISECONDS 500
constexpr std::string_view testingImagePathsFile = "../imagesTestList.txt";

struct NBL_CAPTION_DATA_TO_DISPLAY
{
	std::string viewType;
	std::string name;
	std::string extension;
};

class ColorSpaceTestSampleApp : public ApplicationBase
{
	constexpr static uint32_t WIN_W = 512u;
	constexpr static uint32_t WIN_H = 512u;
	constexpr static uint32_t SC_IMG_COUNT = 3u;
	constexpr static uint32_t FRAMES_IN_FLIGHT = 5u;
	static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

	core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
	core::smart_refctd_ptr<nbl::ui::IWindow> window;
	core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
	core::smart_refctd_ptr<nbl::video::IAPIConnection> apiConnection;
	core::smart_refctd_ptr<nbl::video::ISurface> surface;
	core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
	core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
	video::IPhysicalDevice* physicalDevice;
	std::array<video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> queues;
	core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
	core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, CommonAPI::InitOutput::MaxSwapChainImageCount> fbos;
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxQueuesCount> commandPools;
	core::smart_refctd_ptr<nbl::system::ISystem> system;
	core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
	video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
	core::smart_refctd_ptr<nbl::system::ILogger> logger;
	core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;
	video::IGPUObjectFromAssetConverter cpu2gpu;

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

	APP_CONSTRUCTOR(ColorSpaceTestSampleApp);

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
			video::EAT_VULKAN,
			"09.ColorSpaceTest",
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
		renderpass = std::move(initOutput.renderpass);
		fbos = std::move(initOutput.fbo);
		commandPools = std::move(initOutput.commandPools);
		assetManager = std::move(initOutput.assetManager);
		cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
		logger = std::move(initOutput.logger);
		inputSystem = std::move(initOutput.inputSystem);

		video::IGPUObjectFromAssetConverter cpu2gpu;

		auto createDescriptorPool = [&](const uint32_t textureCount)
		{
			constexpr uint32_t maxItemCount = 256u;
			{
				nbl::video::IDescriptorPool::SDescriptorPoolSize poolSize;
				poolSize.count = textureCount;
				poolSize.type = nbl::asset::EDT_COMBINED_IMAGE_SAMPLER;
				return logicalDevice->createDescriptorPool(static_cast<nbl::video::IDescriptorPool::E_CREATE_FLAGS>(0), maxItemCount, 1u, &poolSize);
			}
		};

		video::IGPUDescriptorSetLayout::SBinding binding{ 0u, nbl::asset::EDT_COMBINED_IMAGE_SAMPLER, 1u, nbl::video::IGPUShader::ESS_FRAGMENT, nullptr };
		auto gpuDescriptorSetLayout3 = logicalDevice->createGPUDescriptorSetLayout(&binding, &binding + 1u);
		auto gpuDescriptorPool = createDescriptorPool(1u); // per single texture
		auto fstProtoPipeline = nbl::ext::FullScreenTriangle::createProtoPipeline(cpu2gpuParams);

		auto createGPUPipeline = [&](asset::IImageView<nbl::asset::ICPUImage>::E_TYPE typeOfImage) -> core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline>
		{
			auto getPathToFragmentShader = [&]() -> std::string
			{
				switch (typeOfImage)
				{
				case nbl::asset::IImageView<nbl::asset::ICPUImage>::ET_2D:
					return "../present2D.frag";
				case nbl::asset::IImageView<nbl::asset::ICPUImage>::ET_2D_ARRAY:
					return "../present2DArray.frag";
				case nbl::asset::IImageView<nbl::asset::ICPUImage>::ET_CUBE_MAP:
					return "../presentCubemap.frag";
				default:
					assert(false);
				}
			};

			auto fs_bundle = assetManager->getAsset(getPathToFragmentShader(), {});
			auto fs_contents = fs_bundle.getContents();
			if (fs_contents.empty())
				assert(false);

			asset::ICPUSpecializedShader* cpuFragmentShader = static_cast<asset::ICPUSpecializedShader*>(fs_contents.begin()->get());

			core::smart_refctd_ptr<video::IGPUSpecializedShader> gpuFragmentShader;
			{
				cpu2gpuParams.beginCommandBuffers();
				auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&cpuFragmentShader, &cpuFragmentShader + 1, cpu2gpuParams);
				cpu2gpuParams.waitForCreationToComplete(false);

				if (!gpu_array.get() || gpu_array->size() < 1u || !(*gpu_array)[0])
					assert(false);

				gpuFragmentShader = (*gpu_array)[0];
			}

			auto gpuPipelineLayout = logicalDevice->createGPUPipelineLayout(nullptr, nullptr, nullptr, nullptr, nullptr, core::smart_refctd_ptr(gpuDescriptorSetLayout3));
			return ext::FullScreenTriangle::createRenderpassIndependentPipeline(logicalDevice.get(), fstProtoPipeline, std::move(gpuFragmentShader), std::move(gpuPipelineLayout));
		};

		auto gpuPipelineFor2D = createGPUPipeline(nbl::asset::IImageView<nbl::asset::ICPUImage>::E_TYPE::ET_2D);
		auto gpuPipelineFor2DArrays = createGPUPipeline(nbl::asset::IImageView<nbl::asset::ICPUImage>::E_TYPE::ET_2D_ARRAY);
		auto gpuPipelineForCubemaps = createGPUPipeline(nbl::asset::IImageView<nbl::asset::ICPUImage>::E_TYPE::ET_CUBE_MAP);

		core::vector<nbl::core::smart_refctd_ptr<nbl::asset::ICPUImageView>> cpuImageViews;
		core::vector<NBL_CAPTION_DATA_TO_DISPLAY> captionTexturesData;
		{
			std::ifstream list(testingImagePathsFile.data());
			if (list.is_open())
			{
				std::string line;
				for (; std::getline(list, line); )
				{
					if (line != "" && line[0] != ';')
					{
						auto& pathToTexture = line;
						auto& newCpuImageViewTexture = cpuImageViews.emplace_back();

						constexpr auto cachingFlags = static_cast<nbl::asset::IAssetLoader::E_CACHING_FLAGS>(nbl::asset::IAssetLoader::ECF_DONT_CACHE_REFERENCES & nbl::asset::IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL);
						nbl::asset::IAssetLoader::SAssetLoadParams loadParams(0ull, nullptr, cachingFlags);
						auto cpuTextureBundle = assetManager->getAsset(pathToTexture, loadParams);
						auto cpuTextureContents = cpuTextureBundle.getContents();
						{
							bool status = !cpuTextureContents.empty();
							assert(status);
						}

						if (cpuTextureContents.begin() == cpuTextureContents.end())
							assert(false); // cannot perform test in this scenario
						
						// Since this is ColorSpaceTest
						const asset::IImage::E_ASPECT_FLAGS aspectMask = asset::IImage::EAF_COLOR_BIT;
						auto asset = *cpuTextureContents.begin();
						switch (asset->getAssetType())
						{
						case nbl::asset::IAsset::ET_IMAGE:
						{
							nbl::asset::ICPUImageView::SCreationParams viewParams = {};
							viewParams.flags = static_cast<decltype(viewParams.flags)>(0u);
							viewParams.image = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(asset);
							viewParams.format = viewParams.image->getCreationParameters().format;
							viewParams.viewType = decltype(viewParams.viewType)::ET_2D;
							viewParams.subresourceRange.aspectMask = aspectMask;
							viewParams.subresourceRange.baseArrayLayer = 0u;
							viewParams.subresourceRange.layerCount = 1u;
							viewParams.subresourceRange.baseMipLevel = 0u;
							viewParams.subresourceRange.levelCount = 1u;

							newCpuImageViewTexture = nbl::asset::ICPUImageView::create(std::move(viewParams));
						} break;

						case nbl::asset::IAsset::ET_IMAGE_VIEW:
						{
							newCpuImageViewTexture = core::smart_refctd_ptr_static_cast<asset::ICPUImageView>(asset);
						} break;

						default:
						{
							assert(false); // in that case provided asset is wrong
						}
						}
						
						newCpuImageViewTexture->getCreationParameters().image->addImageUsageFlags(asset::IImage::EUF_SAMPLED_BIT);

						std::filesystem::path filename, extension;
						core::splitFilename(pathToTexture.c_str(), nullptr, &filename, &extension);

						auto& captionData = captionTexturesData.emplace_back();
						captionData.name = filename.string();
						captionData.extension = extension.string();
						captionData.viewType = [&]()
						{
							const auto& viewType = newCpuImageViewTexture->getCreationParameters().viewType;

							if (viewType == nbl::asset::IImageView<nbl::video::IGPUImage>::ET_2D)
								return std::string("ET_2D");
							else if (viewType == nbl::asset::IImageView<nbl::video::IGPUImage>::ET_2D_ARRAY)
								return std::string("ET_2D_ARRAY");
							else if (viewType == nbl::asset::IImageView<nbl::video::IGPUImage>::ET_CUBE_MAP)
								return std::string("ET_CUBE_MAP");
							else
								assert(false);
						}();

						const std::string finalFileNameWithExtension = captionData.name + captionData.extension;
						std::cout << finalFileNameWithExtension << "\n";

						auto tryToWrite = [&](asset::IAsset* asset)
						{
							asset::IAssetWriter::SAssetWriteParams wparams(asset);
							std::string assetPath = "imageAsset_" + finalFileNameWithExtension;
							return assetManager->writeAsset(assetPath, wparams);
						};

						if (!tryToWrite(newCpuImageViewTexture->getCreationParameters().image.get()))
							if (!tryToWrite(newCpuImageViewTexture.get()))
								assert(false); // could not write an asset
					}
				}
			}
		}

		cpu2gpuParams.beginCommandBuffers();
		auto gpuImageViews = cpu2gpu.getGPUObjectsFromAssets(cpuImageViews.data(), cpuImageViews.data() + cpuImageViews.size(), cpu2gpuParams);
		cpu2gpuParams.waitForCreationToComplete(false);

		if (!gpuImageViews || gpuImageViews->size() < cpuImageViews.size())
			assert(false);

		auto getCurrentGPURenderpassIndependentPipeline = [&](nbl::video::IGPUImageView* gpuImageView)
		{
			switch (gpuImageView->getCreationParameters().viewType)
			{
			case nbl::asset::IImageView<nbl::video::IGPUImage>::ET_2D:
			{
				return gpuPipelineFor2D;
			}

			case nbl::asset::IImageView<nbl::video::IGPUImage>::ET_2D_ARRAY:
			{
				return gpuPipelineFor2DArrays;
			}

			case nbl::asset::IImageView<nbl::video::IGPUImage>::ET_CUBE_MAP:
			{
				return gpuPipelineForCubemaps;
			}

			default:
				assert(false);
			}
		};

		auto presentImageOnTheScreen = [&](nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> gpuImageView, const NBL_CAPTION_DATA_TO_DISPLAY& captionData)
		{
			auto ds = logicalDevice->createGPUDescriptorSet(gpuDescriptorPool.get(), nbl::core::smart_refctd_ptr(gpuDescriptorSetLayout3));

			nbl::video::IGPUDescriptorSet::SDescriptorInfo info;
			{
				info.desc = gpuImageView;
				nbl::asset::ISampler::SParams samplerParams = { nbl::asset::ISampler::ETC_CLAMP_TO_EDGE, nbl::asset::ISampler::ETC_CLAMP_TO_EDGE, nbl::asset::ISampler::ETC_CLAMP_TO_EDGE, nbl::asset::ISampler::ETBC_FLOAT_OPAQUE_BLACK, nbl::asset::ISampler::ETF_LINEAR, nbl::asset::ISampler::ETF_LINEAR, nbl::asset::ISampler::ESMM_LINEAR, 0u, false, nbl::asset::ECO_ALWAYS };
				info.image.sampler = logicalDevice->createGPUSampler(samplerParams);
				info.image.imageLayout = nbl::asset::EIL_SHADER_READ_ONLY_OPTIMAL;
			}

			nbl::video::IGPUDescriptorSet::SWriteDescriptorSet write;
			write.dstSet = ds.get();
			write.binding = 0u;
			write.arrayElement = 0u;
			write.count = 1u;
			write.descriptorType = nbl::asset::EDT_COMBINED_IMAGE_SAMPLER;
			write.info = &info;

			logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);

			auto currentGpuRenderpassIndependentPipeline = getCurrentGPURenderpassIndependentPipeline(gpuImageView.get());
			core::smart_refctd_ptr<nbl::video::IGPUGraphicsPipeline> gpuGraphicsPipeline;
			{
				nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams = {};
				graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(const_cast<video::IGPURenderpassIndependentPipeline*>(currentGpuRenderpassIndependentPipeline.get()));
				graphicsPipelineParams.renderpass = core::smart_refctd_ptr(renderpass);

				gpuGraphicsPipeline = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));
			}

			const std::string windowCaption = "[Nabla Engine] Color Space Test Demo - CURRENT IMAGE: " + captionData.name + " - VIEW TYPE: " + captionData.viewType + " - EXTENSION: " + captionData.extension;
			window->setCaption(windowCaption);

			core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FRAMES_IN_FLIGHT];
			logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, commandBuffers);

			core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
			core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
			core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };

			for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
			{
				imageAcquire[i] = logicalDevice->createSemaphore();
				renderFinished[i] = logicalDevice->createSemaphore();
			}

			auto startPoint = std::chrono::high_resolution_clock::now();

			uint32_t imgnum = 0u;
			int32_t resourceIx = -1;
			for (;;)
			{
				resourceIx++;
				if (resourceIx >= FRAMES_IN_FLIGHT)
					resourceIx = 0;

				auto& cb = commandBuffers[resourceIx];
				auto& fence = frameComplete[resourceIx];
				if (fence)
				{
					while (logicalDevice->waitForFences(1u, &fence.get(), false, MAX_TIMEOUT) == video::IGPUFence::ES_TIMEOUT)
					{
					}
					logicalDevice->resetFences(1u, &fence.get());
				}
				else
					fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

				auto aPoint = std::chrono::high_resolution_clock::now();
				if (std::chrono::duration_cast<std::chrono::milliseconds>(aPoint - startPoint).count() > SWITCH_IMAGES_PER_X_MILISECONDS)
					break;

				// acquire image 
				swapchain->acquireNextImage(MAX_TIMEOUT, imageAcquire[resourceIx].get(), nullptr, &imgnum);

				cb->begin(0);

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

				video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
				{
					VkRect2D area;
					area.offset = { 0,0 };
					area.extent = { WIN_W, WIN_H };
					asset::SClearValue clear;
					clear.color.float32[0] = 1.f;
					clear.color.float32[1] = 1.f;
					clear.color.float32[2] = 1.f;
					clear.color.float32[3] = 1.f;
					beginInfo.clearValueCount = 1u;
					beginInfo.framebuffer = fbos[imgnum];
					beginInfo.renderpass = renderpass;
					beginInfo.renderArea = area;
					beginInfo.clearValues = &clear;
				}

				cb->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);
				cb->bindGraphicsPipeline(gpuGraphicsPipeline.get());
				cb->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuGraphicsPipeline->getRenderpassIndependentPipeline()->getLayout(), 3, 1, &ds.get(), nullptr);
				ext::FullScreenTriangle::recordDrawCalls(cb.get());
				cb->endRenderPass();
				cb->end();

				CommonAPI::Submit(
					logicalDevice.get(),
					swapchain.get(),
					cb.get(),
					queues[CommonAPI::InitOutput::EQT_GRAPHICS],
					imageAcquire[resourceIx].get(),
					renderFinished[resourceIx].get(),
					fence.get());

				CommonAPI::Present(
					logicalDevice.get(),
					swapchain.get(),
					queues[CommonAPI::InitOutput::EQT_GRAPHICS],
					renderFinished[resourceIx].get(),
					imgnum);
			}

			logicalDevice->waitIdle();

			const auto& fboCreationParams = fbos[imgnum]->getCreationParameters();
			auto gpuSourceImageView = fboCreationParams.attachments[0];

			const std::string writePath = "screenShot_" + captionData.name + ".png";

			return ext::ScreenShot::createScreenShot(
				logicalDevice.get(),
				queues[decltype(initOutput)::EQT_TRANSFER_UP],
				nullptr,
				gpuSourceImageView.get(),
				assetManager.get(),
				writePath,
				asset::EIL_PRESENT_SRC,
				static_cast<asset::E_ACCESS_FLAGS>(0u));
		};

		for (size_t i = 0; i < gpuImageViews->size(); ++i)
		{
			auto gpuImageView = (*gpuImageViews)[i];
			if (gpuImageView)
			{
				auto& captionData = captionTexturesData[i];

				bool status = presentImageOnTheScreen(nbl::core::smart_refctd_ptr(gpuImageView), captionData);
				assert(status);
			}
		}
	}

	void onAppTerminated_impl() override
	{

	}

	void workLoopBody() override
	{
		
	}

	bool keepRunning() override
	{
		return false;
	}
};

NBL_COMMON_API_MAIN(ColorSpaceTestSampleApp)

extern "C" {  _declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; }