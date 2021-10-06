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
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"

using namespace nbl;
using namespace core;

class ColorSpaceTestApp : public ApplicationBase
{
#define SWITCH_IMAGES_PER_X_MILISECONDS 500
	static constexpr std::string_view testingImagePathsFile = "../imagesTestList.txt";

	struct NBL_CAPTION_DATA_TO_DISPLAY
	{
		std::string viewType;
		std::string name;
		std::string extension;
	};

	static constexpr uint32_t NBL_WINDOW_WIDTH = 1280;
	static constexpr uint32_t NBL_WINDOW_HEIGHT = 720;
	static constexpr uint32_t SC_IMG_COUNT = 3u;
	static constexpr uint32_t FRAMES_IN_FLIGHT = 5u;
	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

public:
	struct Nabla : IUserData
	{
		// TODO: check for unused members
		core::smart_refctd_ptr<ui::IWindowManager> windowManager;
		core::smart_refctd_ptr<ui::IWindow> window;
		core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
		core::smart_refctd_ptr<video::IAPIConnection> gl;
		core::smart_refctd_ptr<video::ISurface> surface;
		core::smart_refctd_ptr<video::IUtilities> utilities;
		core::smart_refctd_ptr<video::ILogicalDevice> logicalDevice;
		video::IPhysicalDevice* gpuPhysicalDevice;
		std::array<video::IGPUQueue*, CommonAPI::InitOutput<SC_IMG_COUNT>::EQT_COUNT> queues = { nullptr, nullptr, nullptr, nullptr };
		core::smart_refctd_ptr<video::ISwapchain> swapchain;
		core::smart_refctd_ptr<video::IGPURenderpass> renderpass;
		std::array<core::smart_refctd_ptr<video::IGPUFramebuffer>, SC_IMG_COUNT> fbos;
		core::smart_refctd_ptr<video::IGPUCommandPool> commandPool;
		core::smart_refctd_ptr<system::ISystem> system;
		core::smart_refctd_ptr<asset::IAssetManager> assetManager;
		video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
		core::smart_refctd_ptr<nbl::system::ILogger> logger;
		core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;
		core::smart_refctd_ptr<video::IDescriptorPool> gpuDescriptorPool;

		core::smart_refctd_ptr<video::IGPUFence> gpuTransferFence;
		core::smart_refctd_ptr<video::IGPUFence> gpuComputeFence;
		video::IGPUObjectFromAssetConverter cpu2gpu;

		core::smart_refctd_ptr<video::IGPUSemaphore> acquireSemaphores[FRAMES_IN_FLIGHT];
		core::smart_refctd_ptr<video::IGPUSemaphore> releaseSemaphores[FRAMES_IN_FLIGHT];
		core::smart_refctd_ptr<video::IGPUFence> frameFences[FRAMES_IN_FLIGHT];

		video::created_gpu_object_array<asset::ICPUImageView> gpuImageViews;
		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> gpuDescriptorSetLayout3;
		nbl::core::vector<NBL_CAPTION_DATA_TO_DISPLAY> captionTexturesData;
		ext::FullScreenTriangle::NBL_PROTO_PIPELINE fstProtoPipeline;

		core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> gpuPipelineFor2D;
		core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> gpuPipelineFor2DArrays;
		core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> gpuPipelineForCubemaps;

		uint32_t imagesPresented = 0u;

		void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override
		{
			window = std::move(wnd);
		}

		void cpu2gpuWaitForFences()
		{
			video::IGPUFence::E_STATUS waitStatus = video::IGPUFence::ES_NOT_READY;
			while (waitStatus != video::IGPUFence::ES_SUCCESS)
			{
				waitStatus = logicalDevice->waitForFences(1u, &gpuTransferFence.get(), false, 999999999ull);
				if (waitStatus == video::IGPUFence::ES_ERROR)
					assert(false);
				else if (waitStatus == video::IGPUFence::ES_TIMEOUT)
					break;
			}

			waitStatus = video::IGPUFence::ES_NOT_READY;
			while (waitStatus != video::IGPUFence::ES_SUCCESS)
			{
				waitStatus = logicalDevice->waitForFences(1u, &gpuComputeFence.get(), false, 999999999ull);
				if (waitStatus == video::IGPUFence::ES_ERROR)
					assert(false);
				else if (waitStatus == video::IGPUFence::ES_TIMEOUT)
					break;
			}
		};

		core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> createGPUPipeline(nbl::asset::IImageView<nbl::asset::ICPUImage>::E_TYPE typeOfImage)
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

			nbl::core::smart_refctd_ptr<video::IGPUSpecializedShader> gpuFragmentShader;
			{
				auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&cpuFragmentShader, &cpuFragmentShader + 1, cpu2gpuParams);
				if (!gpu_array.get() || gpu_array->size() < 1u || !(*gpu_array)[0])
					assert(false);

				cpu2gpuWaitForFences();
				gpuFragmentShader = (*gpu_array)[0];
			}

			auto gpuPipelineLayout = logicalDevice->createGPUPipelineLayout(nullptr, nullptr, nullptr, nullptr, nullptr, core::smart_refctd_ptr(gpuDescriptorSetLayout3));
			return ext::FullScreenTriangle::createRenderpassIndependentPipeline(logicalDevice.get(), fstProtoPipeline, std::move(gpuFragmentShader), std::move(gpuPipelineLayout));
		}

		core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> getCurrentGPURenderpassIndependentPipeline(nbl::video::IGPUImageView* gpuImageView)
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
		}

		bool presentImageOnTheScreen(nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> gpuImageView, const NBL_CAPTION_DATA_TO_DISPLAY& captionData)
		{
			auto gpuSamplerDescriptorSet3 = logicalDevice->createGPUDescriptorSet(gpuDescriptorPool.get(), nbl::core::smart_refctd_ptr(gpuDescriptorSetLayout3));

			nbl::video::IGPUDescriptorSet::SDescriptorInfo info;
			{
				info.desc = gpuImageView;
				nbl::asset::ISampler::SParams samplerParams = { nbl::asset::ISampler::ETC_CLAMP_TO_EDGE, nbl::asset::ISampler::ETC_CLAMP_TO_EDGE, nbl::asset::ISampler::ETC_CLAMP_TO_EDGE, nbl::asset::ISampler::ETBC_FLOAT_OPAQUE_BLACK, nbl::asset::ISampler::ETF_LINEAR, nbl::asset::ISampler::ETF_LINEAR, nbl::asset::ISampler::ESMM_LINEAR, 0u, false, nbl::asset::ECO_ALWAYS };
				info.image.sampler = logicalDevice->createGPUSampler(samplerParams);
				info.image.imageLayout = nbl::asset::EIL_SHADER_READ_ONLY_OPTIMAL;
			}

			nbl::video::IGPUDescriptorSet::SWriteDescriptorSet write;
			write.dstSet = gpuSamplerDescriptorSet3.get();
			write.binding = 0u;
			write.arrayElement = 0u;
			write.count = 1u;
			write.descriptorType = nbl::asset::EDT_COMBINED_IMAGE_SAMPLER;
			write.info = &info;

			logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);

			auto currentGpuRenderpassIndependentPipeline = getCurrentGPURenderpassIndependentPipeline(gpuImageView.get());
			nbl::core::smart_refctd_ptr<nbl::video::IGPUGraphicsPipeline> gpuGraphicsPipeline;
			{
				nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
				graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(const_cast<video::IGPURenderpassIndependentPipeline*>(currentGpuRenderpassIndependentPipeline.get()));
				graphicsPipelineParams.renderpass = core::smart_refctd_ptr(renderpass);

				gpuGraphicsPipeline = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));
			}

			const std::string windowCaption = "[Nabla Engine] Color Space Test Demo - CURRENT IMAGE: " + captionData.name + " - VIEW TYPE: " + captionData.viewType + " - EXTENSION: " + captionData.extension;
			window->setCaption(windowCaption);

			core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FRAMES_IN_FLIGHT];
			logicalDevice->createCommandBuffers(commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, commandBuffers);

			core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
			core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
			core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };

			for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
			{
				imageAcquire[i] = logicalDevice->createSemaphore();
				renderFinished[i] = logicalDevice->createSemaphore();
			}

			auto startPoint = std::chrono::high_resolution_clock::now();

			constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
			uint32_t acquiredNextFBO = {};
			auto resourceIx = -1;

			while (true)
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

				auto aPoint = std::chrono::high_resolution_clock::now();
				if (std::chrono::duration_cast<std::chrono::milliseconds>(aPoint - startPoint).count() > SWITCH_IMAGES_PER_X_MILISECONDS)
					break;

				commandBuffer->reset(nbl::video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
				commandBuffer->begin(0);

				asset::SViewport viewport;
				viewport.minDepth = 1.f;
				viewport.maxDepth = 0.f;
				viewport.x = 0u;
				viewport.y = 0u;
				viewport.width = NBL_WINDOW_WIDTH;
				viewport.height = NBL_WINDOW_HEIGHT;
				commandBuffer->setViewport(0u, 1u, &viewport);

				swapchain->acquireNextImage(MAX_TIMEOUT, imageAcquire[resourceIx].get(), nullptr, &acquiredNextFBO);

				nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
				{
					VkRect2D area;
					area.offset = { 0,0 };
					area.extent = { NBL_WINDOW_WIDTH, NBL_WINDOW_HEIGHT };
					nbl::asset::SClearValue clear;
					clear.color.float32[0] = 1.f;
					clear.color.float32[1] = 1.f;
					clear.color.float32[2] = 1.f;
					clear.color.float32[3] = 1.f;
					beginInfo.clearValueCount = 1u;
					beginInfo.framebuffer = fbos[acquiredNextFBO];
					beginInfo.renderpass = renderpass;
					beginInfo.renderArea = area;
					beginInfo.clearValues = &clear;
				}

				commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);
				commandBuffer->bindGraphicsPipeline(gpuGraphicsPipeline.get());
				commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuGraphicsPipeline->getRenderpassIndependentPipeline()->getLayout(), 3, 1, &gpuSamplerDescriptorSet3.get(), nullptr);
				ext::FullScreenTriangle::recordDrawCalls(commandBuffer.get());
				commandBuffer->endRenderPass();
				commandBuffer->end();

				CommonAPI::Submit(logicalDevice.get(), swapchain.get(), commandBuffer.get(), queues[CommonAPI::InitOutput<1>::EQT_GRAPHICS], imageAcquire[resourceIx].get(), renderFinished[resourceIx].get(), fence.get());
				CommonAPI::Present(logicalDevice.get(), swapchain.get(), queues[CommonAPI::InitOutput<1>::EQT_GRAPHICS], renderFinished[resourceIx].get(), acquiredNextFBO);
			}

			const auto& fboCreationParams = fbos[acquiredNextFBO]->getCreationParameters();
			auto gpuSourceImageView = fboCreationParams.attachments[0];

			const std::string writePath = "screenShot_" + captionData.name + ".png";
			bool status = ext::ScreenShot::createScreenShot(logicalDevice.get(), queues[CommonAPI::InitOutput<1>::EQT_TRANSFER_UP], renderFinished[resourceIx].get(), gpuSourceImageView.get(), assetManager.get(), writePath);
			return status;
		}
	};

	APP_CONSTRUCTOR(ColorSpaceTestApp)

	void onAppInitialized_impl(void* data) override
	{
		Nabla* engine = static_cast<Nabla*>(data);

		CommonAPI::InitOutput<SC_IMG_COUNT> initOutput;
		initOutput.window = core::smart_refctd_ptr(engine->window);
		CommonAPI::Init<NBL_WINDOW_WIDTH, NBL_WINDOW_HEIGHT, SC_IMG_COUNT>(initOutput, video::EAT_OPENGL, "MeshLoaders", nbl::asset::EF_D32_SFLOAT);
		engine->window = std::move(initOutput.window);
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
		engine->cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
		engine->utilities = std::move(initOutput.utilities);

		engine->gpuTransferFence = engine->logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
		engine->gpuComputeFence = engine->logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

		nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
		{
			engine->cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].fence = &engine->gpuTransferFence;
			engine->cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].fence = &engine->gpuComputeFence;
		}

		auto createDescriptorPool = [&](const uint32_t textureCount)
		{
			constexpr uint32_t maxItemCount = 256u;
			{
				nbl::video::IDescriptorPool::SDescriptorPoolSize poolSize;
				poolSize.count = textureCount;
				poolSize.type = nbl::asset::EDT_COMBINED_IMAGE_SAMPLER;
				return engine->logicalDevice->createDescriptorPool(static_cast<nbl::video::IDescriptorPool::E_CREATE_FLAGS>(0), maxItemCount, 1u, &poolSize);
			}
		};

		nbl::video::IGPUDescriptorSetLayout::SBinding binding{ 0u, nbl::asset::EDT_COMBINED_IMAGE_SAMPLER, 1u, nbl::video::IGPUSpecializedShader::ESS_FRAGMENT, nullptr };
		engine->gpuDescriptorSetLayout3 = engine->logicalDevice->createGPUDescriptorSetLayout(&binding, &binding + 1u);
		engine->gpuDescriptorPool = createDescriptorPool(1u); // per single texture
		engine->fstProtoPipeline = nbl::ext::FullScreenTriangle::createProtoPipeline(engine->cpu2gpuParams);
		
		

		engine->gpuPipelineFor2D = engine->createGPUPipeline(nbl::asset::IImageView<nbl::asset::ICPUImage>::E_TYPE::ET_2D);
		engine->gpuPipelineFor2DArrays = engine->createGPUPipeline(nbl::asset::IImageView<nbl::asset::ICPUImage>::E_TYPE::ET_2D_ARRAY);
		engine->gpuPipelineForCubemaps = engine->createGPUPipeline(nbl::asset::IImageView<nbl::asset::ICPUImage>::E_TYPE::ET_CUBE_MAP);

		nbl::core::vector<nbl::core::smart_refctd_ptr<nbl::asset::ICPUImageView>> cpuImageViews;
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
						auto cpuTextureBundle = engine->assetManager->getAsset(pathToTexture, loadParams);
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
							nbl::asset::ICPUImageView::SCreationParams viewParams;
							viewParams.flags = static_cast<decltype(viewParams.flags)>(0u);
							viewParams.image = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(asset);
							viewParams.format = viewParams.image->getCreationParameters().format;
							viewParams.viewType = decltype(viewParams.viewType)::ET_2D;
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

						auto& captionData = engine->captionTexturesData.emplace_back();
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
							return engine->assetManager->writeAsset(assetPath, wparams);
						};

						if (!tryToWrite(newCpuImageViewTexture->getCreationParameters().image.get()))
							if (!tryToWrite(newCpuImageViewTexture.get()))
								assert(false); // could not write an asset
					}
				}
			}
		}

		engine->gpuImageViews = cpu2gpu.getGPUObjectsFromAssets(cpuImageViews.data(), cpuImageViews.data() + cpuImageViews.size(), engine->cpu2gpuParams);
		if (!engine->gpuImageViews || engine->gpuImageViews->size() < cpuImageViews.size())
			assert(false);
		engine->cpu2gpuWaitForFences();
	}

	void onAppTerminated_impl(void* data) override
	{
		Nabla* engine = static_cast<Nabla*>(data);
	}

	void workLoopBody(void* data) override
	{
		Nabla* engine = static_cast<Nabla*>(data);
		auto gpuImageView = (*engine->gpuImageViews)[engine->imagesPresented];
		auto& captionData = engine->captionTexturesData[engine->imagesPresented];

		bool status = engine->presentImageOnTheScreen(nbl::core::smart_refctd_ptr(gpuImageView), captionData);
		assert(status);

		engine->imagesPresented++;
	}

	bool keepRunning(void* params) override
	{
		Nabla* engine = static_cast<Nabla*>(params);
		return engine->imagesPresented < engine->gpuImageViews->size();
	}

};

NBL_COMMON_API_MAIN(ColorSpaceTestApp, ColorSpaceTestApp::Nabla)