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
using namespace core;

#define SWITCH_IMAGES_PER_X_MILISECONDS 500
constexpr std::string_view testingImagePathsFile = "../imagesTestList.txt";

struct NBL_CAPTION_DATA_TO_DISPLAY
{
	std::string viewType;
	std::string name;
	std::string extension;
};

int main()
{
    constexpr uint32_t WINDOW_WIDTH = 1280;
    constexpr uint32_t WINDOW_HEIGHT = 720;
	constexpr uint32_t SC_IMG_COUNT = 3u;
	constexpr uint32_t FRAMES_IN_FLIGHT = 2u;
	// static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

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

    auto initOutput = CommonAPI::Init<WINDOW_WIDTH, WINDOW_HEIGHT, SC_IMG_COUNT>(
		video::EAT_VULKAN,
		"09.ColorSpaceTest",
		requiredInstanceFeatures,
		optionalInstanceFeatures,
		requiredDeviceFeatures,
		optionalDeviceFeatures,
		swapchainImageUsage,
		surfaceFormat);

    auto window = std::move(initOutput.window);
    auto gl = std::move(initOutput.apiConnection);
    auto surface = std::move(initOutput.surface);
    auto gpuPhysicalDevice = std::move(initOutput.physicalDevice);
    auto logicalDevice = std::move(initOutput.logicalDevice);
    auto queues = std::move(initOutput.queues);
    auto swapchain = std::move(initOutput.swapchain);
    auto renderpass = std::move(initOutput.renderpass);
    auto fbos = std::move(initOutput.fbo);
    auto commandPool = std::move(initOutput.commandPool);
    auto assetManager = std::move(initOutput.assetManager);
	auto cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
	auto utilities = std::move(initOutput.utilities);

	nbl::video::IGPUObjectFromAssetConverter cpu2gpu;

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

	nbl::video::IGPUDescriptorSetLayout::SBinding binding{ 0u, nbl::asset::EDT_COMBINED_IMAGE_SAMPLER, 1u, nbl::video::IGPUShader::ESS_FRAGMENT, nullptr };
	auto gpuDescriptorSetLayout3 = logicalDevice->createGPUDescriptorSetLayout(&binding, &binding + 1u);
	auto gpuDescriptorPool = createDescriptorPool(1u); // per single texture
	auto fstProtoPipeline = nbl::ext::FullScreenTriangle::createProtoPipeline(cpu2gpuParams);
	
	auto createGPUPipeline = [&](nbl::asset::IImageView<nbl::asset::ICPUImage>::E_TYPE typeOfImage) -> core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline>
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
				{
					assert(false);
				}
			}
		};

		auto fs_bundle = assetManager->getAsset(getPathToFragmentShader(), {});
		auto fs_contents = fs_bundle.getContents();
		if (fs_contents.empty())
			assert(false);

		asset::ICPUSpecializedShader* cpuFragmentShader = static_cast<nbl::asset::ICPUSpecializedShader*>(fs_contents.begin()->get());
		
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

					auto asset = *cpuTextureContents.begin();
					switch (asset->getAssetType())
					{
						case nbl::asset::IAsset::ET_IMAGE:
						{
							// Since this is ColorSpaceTest
							const asset::IImage::E_ASPECT_FLAGS aspectMask = asset::IImage::EAF_COLOR_BIT;

							nbl::asset::ICPUImageView::SCreationParams viewParams = {};
							viewParams.flags = static_cast<decltype(viewParams.flags)>(0u);
							viewParams.image = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(asset);
							const auto newUsageFlags = viewParams.image->getImageUsageFlags() | asset::IImage::EUF_COLOR_ATTACHMENT_BIT | asset::IImage::EUF_SAMPLED_BIT;
							viewParams.image->setImageUsageFlags(newUsageFlags);
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

	core::smart_refctd_ptr<video::IGPUCommandBuffer> transferCmdBuffer, computeCmdBuffer;

	logicalDevice->createCommandBuffers(commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &transferCmdBuffer);
	logicalDevice->createCommandBuffers(commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &computeCmdBuffer);

	cpu2gpuParams.perQueue[video::IGPUObjectFromAssetConverter::EQU_TRANSFER].cmdbuf = transferCmdBuffer;
	cpu2gpuParams.perQueue[video::IGPUObjectFromAssetConverter::EQU_COMPUTE].cmdbuf = computeCmdBuffer;
	
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
			{
				assert(false);
			}
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

		core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[SC_IMG_COUNT];
		logicalDevice->createCommandBuffers(commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, SC_IMG_COUNT, commandBuffers);

		core::smart_refctd_ptr<video::IGPUFence> frameFences[FRAMES_IN_FLIGHT] = { nullptr };
		core::smart_refctd_ptr<video::IGPUSemaphore> acquireSemaphores[FRAMES_IN_FLIGHT] = { nullptr };
		core::smart_refctd_ptr<video::IGPUSemaphore> releaseSemaphores[FRAMES_IN_FLIGHT] = { nullptr };

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
		{
			acquireSemaphores[i] = logicalDevice->createSemaphore();
			releaseSemaphores[i] = logicalDevice->createSemaphore();
			frameFences[i] = logicalDevice->createFence(video::IGPUFence::E_CREATE_FLAGS::ECF_SIGNALED_BIT);
		}

		const uint32_t swapchainImageCount = swapchain->getImageCount();
		for (uint32_t i = 0u; i < swapchainImageCount; ++i)
		{
			commandBuffers[i]->begin(0);

			asset::SViewport viewport;
			viewport.minDepth = 1.f;
			viewport.maxDepth = 0.f;
			viewport.x = 0u;
			viewport.y = 0u;
			viewport.width = WINDOW_WIDTH;
			viewport.height = WINDOW_HEIGHT;
			commandBuffers[i]->setViewport(0u, 1u, &viewport);

			nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
			{
				VkRect2D area;
				area.offset = { 0,0 };
				area.extent = { WINDOW_WIDTH, WINDOW_HEIGHT };
				nbl::asset::SClearValue clear;
				clear.color.float32[0] = 1.f;
				clear.color.float32[1] = 1.f;
				clear.color.float32[2] = 1.f;
				clear.color.float32[3] = 1.f;
				beginInfo.clearValueCount = 1u;
				beginInfo.framebuffer = fbos[i];
				beginInfo.renderpass = renderpass;
				beginInfo.renderArea = area;
				beginInfo.clearValues = &clear;
			}

			commandBuffers[i]->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);
			commandBuffers[i]->bindGraphicsPipeline(gpuGraphicsPipeline.get());
			commandBuffers[i]->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuGraphicsPipeline->getRenderpassIndependentPipeline()->getLayout(), 0, 1, &ds.get(), nullptr);
			ext::FullScreenTriangle::recordDrawCalls(commandBuffers[i].get());
			commandBuffers[i]->endRenderPass();
			commandBuffers[i]->end();
		}

		auto startPoint = std::chrono::high_resolution_clock::now();

		uint32_t currentFrameIndex = 0u;
		uint32_t imageIndex;
		for (;;)
		{
			video::IGPUSemaphore* acquireSemaphore_frame = acquireSemaphores[currentFrameIndex].get();
			video::IGPUSemaphore* releaseSemaphore_frame = releaseSemaphores[currentFrameIndex].get();
			video::IGPUFence* fence_frame = frameFences[currentFrameIndex].get();

			video::IGPUFence::E_STATUS retval = logicalDevice->waitForFences(1u, &fence_frame, true, ~0ull);
			assert(retval == video::IGPUFence::ES_SUCCESS);

			auto aPoint = std::chrono::high_resolution_clock::now();
			if (std::chrono::duration_cast<std::chrono::milliseconds>(aPoint - startPoint).count() > SWITCH_IMAGES_PER_X_MILISECONDS)
				break;

			swapchain->acquireNextImage(~0ull, acquireSemaphore_frame, nullptr, &imageIndex);

			logicalDevice->resetFences(1u, &fence_frame);

			CommonAPI::Submit(
				logicalDevice.get(),
				swapchain.get(),
				commandBuffers[imageIndex].get(),
				queues[CommonAPI::InitOutput<SC_IMG_COUNT>::EQT_GRAPHICS],
				acquireSemaphore_frame,
				releaseSemaphore_frame,
				fence_frame);

			CommonAPI::Present(
				logicalDevice.get(),
				swapchain.get(),
				queues[CommonAPI::InitOutput<SC_IMG_COUNT>::EQT_GRAPHICS],
				releaseSemaphore_frame,
				imageIndex);

			currentFrameIndex = (currentFrameIndex + 1) % FRAMES_IN_FLIGHT;
		}

		logicalDevice->waitIdle();

#if 1
		const auto& fboCreationParams = fbos[imageIndex]->getCreationParameters();
		auto gpuSourceImageView = fboCreationParams.attachments[0];

		const std::string writePath = "screenShot_" + captionData.name + ".png";

		return ext::ScreenShot::createScreenShot(
			logicalDevice.get(),
			queues[decltype(initOutput)::EQT_TRANSFER_UP],
			nullptr,
			gpuSourceImageView.get(),
			assetManager.get(),
			writePath,
			asset::EIL_PRESENT_SRC_KHR);
#endif
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

	return 0;
}