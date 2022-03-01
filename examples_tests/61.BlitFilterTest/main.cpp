// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"

using namespace nbl;
using namespace nbl::asset;
using namespace nbl::core;
using namespace nbl::video;

#define FATAL_LOG(x, ...) {logger->log(##x, system::ILogger::ELL_ERROR, __VA_ARGS__); exit(-1);}

using ScaledBoxKernel = asset::CScaledImageFilterKernel<CBoxImageFilterKernel>;
using BlitFilter = asset::CBlitImageFilter<asset::VoidSwizzle, asset::IdentityDither, void, false, ScaledBoxKernel, ScaledBoxKernel, ScaledBoxKernel>;

core::smart_refctd_ptr<ICPUImage> createCPUImage(const std::array<uint32_t, 2>& dims)
{
	IImage::SCreationParams imageParams = {};
	imageParams.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);
	imageParams.type = IImage::ET_1D;
	imageParams.format = asset::EF_R32_SFLOAT;
	imageParams.extent = { dims[0], dims[1], 1 };
	imageParams.mipLevels = 1u;
	imageParams.arrayLayers = 1u;
	imageParams.samples = asset::ICPUImage::ESCF_1_BIT;

	auto imageRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::IImage::SBufferCopy>>(1ull);
	auto& region = (*imageRegions)[0];
	region.bufferImageHeight = 0u;
	region.bufferOffset = 0ull;
	region.bufferRowLength = dims[0];
	region.imageExtent = { dims[0], dims[1], 1u };
	region.imageOffset = { 0u, 0u, 0u };
	region.imageSubresource.baseArrayLayer = 0u;
	region.imageSubresource.layerCount = 1u;
	region.imageSubresource.mipLevel = 0;

	size_t bufferSize = asset::getTexelOrBlockBytesize(imageParams.format) * region.imageExtent.width * region.imageExtent.height;
	auto imageBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(bufferSize);
	core::smart_refctd_ptr<ICPUImage> image = ICPUImage::create(std::move(imageParams));
	image->setBufferAndRegions(core::smart_refctd_ptr(imageBuffer), imageRegions);

	return image;
}

class BlitFilterTestApp : public ApplicationBase
{
	constexpr static uint32_t SC_IMG_COUNT = 3u;
	constexpr static uint64_t MAX_TIMEOUT = 99999999999999ull;

public:
	void onAppInitialized_impl() override
	{
		CommonAPI::InitOutput initOutput;
		CommonAPI::InitWithNoExt(initOutput, video::EAT_VULKAN, "BlitFilterTest");

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
		commandPools = std::move(initOutput.commandPools);
		assetManager = std::move(initOutput.assetManager);
		cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
		logger = std::move(initOutput.logger);
		inputSystem = std::move(initOutput.inputSystem);

		std::array<uint32_t, 2> inImageDim = { 17*17, 1u }; // { 800u, 5u };
		std::array<uint32_t, 2> outImageDim = { 17, 1u }; // { 16u, 69u };

		auto inImage = createCPUImage(inImageDim);

		float k = 1.f;
		float* inImagePixel = reinterpret_cast<float*>(inImage->getBuffer()->getPointer());
		for (uint32_t j = 0u; j < inImageDim[1]; ++j)
		{
			for (uint32_t i = 0; i < inImageDim[0]; ++i)
				inImagePixel[j * inImageDim[0] + i] = k++;
		}

		const core::vectorSIMDf scaleX(1.f, 1.f, 1.f, 1.f);
		const core::vectorSIMDf scaleY(1.f, 1.f, 1.f, 1.f);
		const core::vectorSIMDf scaleZ(1.f, 1.f, 1.f, 1.f);


		// CPU blit
		core::vector<float> cpuOutput(outImageDim[0] * outImageDim[1]);
		{
			core::smart_refctd_ptr<ICPUImage> outImage = createCPUImage(outImageDim);

			auto kernelX = ScaledBoxKernel(scaleX, asset::CBoxImageFilterKernel()); // (-1/2, 1/2)
			auto kernelY = ScaledBoxKernel(scaleY, asset::CBoxImageFilterKernel()); // (-1/2, 1/2)
			auto kernelZ = ScaledBoxKernel(scaleZ, asset::CBoxImageFilterKernel()); // (-1/2, 1/2)
			BlitFilter::state_type blitFilterState(std::move(kernelX), std::move(kernelY), std::move(kernelZ));

			blitFilterState.inOffsetBaseLayer = core::vectorSIMDu32();
			blitFilterState.inExtentLayerCount = core::vectorSIMDu32(0u, 0u, 0u, inImage->getCreationParameters().arrayLayers) + inImage->getMipSize();
			blitFilterState.inImage = inImage.get();

			blitFilterState.outOffsetBaseLayer = core::vectorSIMDu32();
			blitFilterState.outExtentLayerCount = core::vectorSIMDu32(0u, 0u, 0u, outImage->getCreationParameters().arrayLayers) + outImage->getMipSize();
			blitFilterState.outImage = outImage.get();

			blitFilterState.axisWraps[0] = asset::ISampler::ETC_CLAMP_TO_EDGE;
			blitFilterState.axisWraps[1] = asset::ISampler::ETC_CLAMP_TO_EDGE;
			blitFilterState.axisWraps[2] = asset::ISampler::ETC_CLAMP_TO_EDGE;
			blitFilterState.borderColor = asset::ISampler::E_TEXTURE_BORDER_COLOR::ETBC_FLOAT_OPAQUE_WHITE;

			blitFilterState.enableLUTUsage = false;

			blitFilterState.scratchMemoryByteSize = BlitFilter::getRequiredScratchByteSize(&blitFilterState);
			blitFilterState.scratchMemory = reinterpret_cast<uint8_t*>(_NBL_ALIGNED_MALLOC(blitFilterState.scratchMemoryByteSize, 32));

			if (!BlitFilter::execute(&blitFilterState))
				printf("Blit filter just shit the bed\n");

			_NBL_ALIGNED_FREE(blitFilterState.scratchMemory);

			// Todo(achal): This needs to change when testing with more complex images
			float* outPixel = reinterpret_cast<float*>(outImage->getBuffer()->getPointer());
			memcpy(cpuOutput.data(), outPixel, cpuOutput.size() * sizeof(float));
		}

		// GPU blit
		core::vector<float> gpuOutput(outImageDim[0] * outImageDim[1]);
		{
			auto kernelX = ScaledBoxKernel(scaleX, asset::CBoxImageFilterKernel()); // (-1/2, 1/2)
			auto kernelY = ScaledBoxKernel(scaleY, asset::CBoxImageFilterKernel()); // (-1/2, 1/2)
			auto kernelZ = ScaledBoxKernel(scaleZ, asset::CBoxImageFilterKernel()); // (-1/2, 1/2)

			core::smart_refctd_ptr<ICPUImage> outImage = createCPUImage(outImageDim);

			constexpr uint32_t WG_DIM = 256u;
			struct push_constants_t
			{
				uint32_t inWidth;
				uint32_t outWidth;

				float negativeSupport;
				float positiveSupport;
				float kernelWeight;

				uint32_t windowDim;
				uint32_t maxLoadIndex;
			};

			core::smart_refctd_ptr<video::IGPUImageView> inImageView = nullptr;
			core::smart_refctd_ptr<video::IGPUImageView> outImageView = nullptr;
			core::smart_refctd_ptr<video::IGPUImage> inImageGPU = nullptr;
			core::smart_refctd_ptr<video::IGPUImage> outImageGPU = nullptr;

			inImage->addImageUsageFlags(asset::ICPUImage::EUF_SAMPLED_BIT);
			outImage->addImageUsageFlags(asset::ICPUImage::EUF_STORAGE_BIT);

			core::smart_refctd_ptr<asset::ICPUImage> tmp[2] = { inImage, outImage };
			cpu2gpuParams.beginCommandBuffers();
			auto gpuArray = cpu2gpu.getGPUObjectsFromAssets(tmp, tmp + 2ull, cpu2gpuParams);
			cpu2gpuParams.waitForCreationToComplete();
			if (!gpuArray || gpuArray->size() < 2u || (!(*gpuArray)[0]))
				FATAL_LOG("Failed to convert CPU images to GPU images\n");

			inImageGPU = gpuArray->begin()[0];
			outImageGPU = gpuArray->begin()[1];

			// do layout transition to GENERAL
			// (I think it might be a good idea to allow the user to change asset::ICPUImage's initialLayout and have the asset converter
			// do the layout transition for them)
			core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf = nullptr;
			logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_COMPUTE].get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cmdbuf);
			video::IGPUCommandBuffer::SImageMemoryBarrier barriers[2] = {};
			barriers[0].oldLayout = asset::EIL_UNDEFINED;
			barriers[0].newLayout = asset::EIL_GENERAL;
			barriers[0].srcQueueFamilyIndex = ~0u;
			barriers[0].dstQueueFamilyIndex = ~0u;
			barriers[0].image = inImageGPU;
			barriers[0].subresourceRange.aspectMask = video::IGPUImage::EAF_COLOR_BIT;
			barriers[0].subresourceRange.levelCount = 1u;
			barriers[0].subresourceRange.layerCount = 1u;

			barriers[1] = barriers[0];
			barriers[1].image = outImageGPU;

			cmdbuf->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
			cmdbuf->pipelineBarrier(asset::EPSF_TOP_OF_PIPE_BIT, asset::EPSF_BOTTOM_OF_PIPE_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 2u, barriers);
			cmdbuf->end();

			auto fence = logicalDevice->createFence(video::IGPUFence::ECF_UNSIGNALED);

			video::IGPUQueue::SSubmitInfo submitInfo = {};
			submitInfo.commandBufferCount = 1u;
			submitInfo.commandBuffers = &cmdbuf.get();
			queues[CommonAPI::InitOutput::EQT_COMPUTE]->submit(1u, &submitInfo, fence.get());
			logicalDevice->blockForFences(1u, &fence.get());

			{
				video::IGPUImageView::SCreationParams creationParams = {};
				creationParams.image = inImageGPU;
				creationParams.viewType = video::IGPUImageView::ET_1D;
				creationParams.format = inImageGPU->getCreationParameters().format;
				creationParams.subresourceRange.aspectMask = video::IGPUImage::EAF_COLOR_BIT;
				creationParams.subresourceRange.layerCount = 1u;
				creationParams.subresourceRange.levelCount = 1u;

				inImageView = logicalDevice->createGPUImageView(std::move(creationParams));
			}

			{
				video::IGPUImageView::SCreationParams creationParams = {};
				creationParams.image = outImageGPU;
				creationParams.viewType = video::IGPUImageView::ET_1D;
				creationParams.format = outImageGPU->getCreationParameters().format;
				creationParams.subresourceRange.aspectMask = video::IGPUImage::EAF_COLOR_BIT;
				creationParams.subresourceRange.layerCount = 1u;
				creationParams.subresourceRange.levelCount = 1u;

				outImageView = logicalDevice->createGPUImageView(std::move(creationParams));
			}

			core::smart_refctd_ptr<video::IGPUSampler> sampler = nullptr;
			{
				video::IGPUSampler::SParams params = { };
				params.TextureWrapU = asset::ISampler::ETC_CLAMP_TO_EDGE;
				params.TextureWrapV = asset::ISampler::ETC_CLAMP_TO_EDGE;
				params.TextureWrapW = asset::ISampler::ETC_CLAMP_TO_EDGE;
				params.BorderColor = asset::ISampler::ETBC_FLOAT_OPAQUE_BLACK;
				params.MinFilter = asset::ISampler::ETF_LINEAR;
				params.MaxFilter = asset::ISampler::ETF_LINEAR;
				params.MipmapMode = asset::ISampler::ESMM_NEAREST;
				params.AnisotropicFilter = 0u;
				params.CompareEnable = 0u;
				params.CompareFunc = asset::ISampler::ECO_ALWAYS;
				sampler = logicalDevice->createGPUSampler(std::move(params));
			}

			constexpr uint32_t DESCRIPTOR_COUNT = 2u; // 3u;
			core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> dsLayout = nullptr;
			{
				video::IGPUDescriptorSetLayout::SBinding bindings[DESCRIPTOR_COUNT] = {};

				// 0. inImage
				// 1. outImage
				// 2. cached kernel weights
				for (uint32_t i = 0u; i < DESCRIPTOR_COUNT; ++i)
				{
					bindings[i].binding = i;
					bindings[i].count = 1u;
					bindings[i].stageFlags = asset::IShader::ESS_COMPUTE;
				}
				bindings[0].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
				bindings[0].samplers = &sampler;

				bindings[1].type = asset::EDT_STORAGE_IMAGE;
				// bindings[2].type = asset::EDT_UNIFORM_BUFFER;

				dsLayout = logicalDevice->createGPUDescriptorSetLayout(bindings, bindings + DESCRIPTOR_COUNT);
			}
			const uint32_t dsCount = 1u;
			auto descriptorPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &dsLayout.get(), &dsLayout.get() + 1ull, &dsCount);

			asset::SPushConstantRange pcRange = {};
			pcRange.stageFlags = asset::IShader::ESS_COMPUTE;
			pcRange.offset = 0u;
			pcRange.size = sizeof(push_constants_t);
			core::smart_refctd_ptr<video::IGPUPipelineLayout> pipelineLayout = logicalDevice->createGPUPipelineLayout(&pcRange, &pcRange + 1ull, core::smart_refctd_ptr(dsLayout));

			core::smart_refctd_ptr<video::IGPUDescriptorSet> ds = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), core::smart_refctd_ptr(dsLayout));
			{
				video::IGPUDescriptorSet::SWriteDescriptorSet writes[DESCRIPTOR_COUNT] = {};
				video::IGPUDescriptorSet::SDescriptorInfo infos[DESCRIPTOR_COUNT] = {};

				for (uint32_t i = 0u; i < DESCRIPTOR_COUNT; ++i)
				{
					writes[i].dstSet = ds.get();
					writes[i].binding = i;
					writes[i].arrayElement = 0u;
					writes[i].count = 1u;
					writes[i].info = &infos[i];
				}

				// inImage
				writes[0].descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
				infos[0].desc = inImageView;
				infos[0].image.imageLayout = asset::EIL_GENERAL;
				infos[0].image.sampler = nullptr;

				// outImage
				writes[1].descriptorType = asset::EDT_STORAGE_IMAGE;
				infos[1].desc = outImageView;
				infos[1].image.imageLayout = asset::EIL_GENERAL;
				infos[1].image.sampler = nullptr;

				// cached weights
				// writes[2].descriptorType = asset::EDT_UNIFORM_BUFFER;
				// infos[2].desc = octreeScratchBuffers[0];
				// infos[2].buffer.offset = 0ull;
				// infos[2].buffer.size = octreeScratchBuffers[0]->getCachedCreationParams().declaredSize;

				logicalDevice->updateDescriptorSets(DESCRIPTOR_COUNT, writes, 0u, nullptr);
			}

			const char* blitCompShaderPath = "../blit.comp";
			core::smart_refctd_ptr<video::IGPUSpecializedShader> compShader = nullptr;
			{
				asset::IAssetLoader::SAssetLoadParams params = {};
				params.logger = logger.get();
				auto loadedShader = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(blitCompShaderPath, params).getContents().begin());

				auto unspecOverridenShader = asset::IGLSLCompiler::createOverridenCopy(loadedShader->getUnspecialized(), "#define _NBL_GLSL_WORKGROUP_SIZE_ %d\n", WG_DIM);

				auto specShader_cpu = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecOverridenShader), asset::ISpecializedShader::SInfo(nullptr, nullptr, "main"));

				auto gpuArray = cpu2gpu.getGPUObjectsFromAssets(&specShader_cpu.get(), &specShader_cpu.get() + 1, cpu2gpuParams);
				if (!gpuArray || gpuArray->size() < 1u || !(*gpuArray)[0])
					FATAL_LOG("Failed to convert CPU specialized shader to GPU specialized shaders!\n");

				compShader = gpuArray->begin()[0];
			}

			auto pipeline = logicalDevice->createGPUComputePipeline(nullptr, std::move(pipelineLayout), std::move(compShader));

			cmdbuf->reset(nbl::video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);

			cmdbuf->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
			cmdbuf->bindComputePipeline(pipeline.get());
			cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, pipeline->getLayout(), 0u, 1u, &ds.get());
			push_constants_t pc = { };
			{
				pc.inWidth = inImageDim[0];
				pc.outWidth = outImageDim[0];
				const float scale = float(inImageDim[0]) / float(outImageDim[0]);
				pc.negativeSupport = -0.5f * scale; // kernelX/Y/Z stores absolute values of support so they won't be helpful here
				pc.positiveSupport = 0.5f * scale;
				pc.kernelWeight = 1.f/scale;
				// I think this formulation is better than ceil(scaledPositiveSupport-scaledNegativeSupport) because if scaledPositiveSupport comes out a tad bit
				// greater than what it should be and if scaledNegativeSupport comes out a tad bit smaller than it should be then the distance between them would
				// become a tad bit greater than it should be and when we take a ceil it'll jump up to the next integer thus giving us 1 more than the actual window
				// size, example: 49x1 -> 7x1.
				// I think (hope) it won't happen with this formulation. 
				pc.windowDim = 1.f*scale; // cannot use kernelX/Y/Z.getWindowSize() here because they haven't been scaled yet for upscaling/downscaling
				// the last pixel of the last window which is used to compute the last output pixel, required for bounds checking, avoids repeated computation in the shader
				const float maxOutputPixelCenter = ((pc.outWidth - 1) + 0.5f) * scale;
				pc.maxLoadIndex = core::floor( (maxOutputPixelCenter-0.5f) + core::abs(pc.positiveSupport) );
			}
			cmdbuf->pushConstants(pipeline->getLayout(), asset::IShader::ESS_COMPUTE, 0u, sizeof(push_constants_t), &pc);
			const uint32_t totalWindowCount = outImageDim[0];
			const uint32_t windowDim = std::ceil(inImageDim[0] / outImageDim[0]);
			const uint32_t windowsPerWG = WG_DIM / windowDim; // we want to make sure that the WG covers a window COMPLETELY even if it means throwing away some invocations
			const uint32_t wgCount = (totalWindowCount + windowsPerWG - 1) / windowsPerWG;
			cmdbuf->dispatch(wgCount, 1u, 1u);

			// after the dispatch download the buffer to check
			core::smart_refctd_ptr<video::IGPUBuffer> downloadBuffer = nullptr;
			{
				const size_t downloadSize = outImageDim[0] * outImageDim[1] * sizeof(float); // Todo(achal): Need to change this for more complex images
				video::IGPUBuffer::SCreationParams creationParams = {};
				creationParams.usage = video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
				downloadBuffer = logicalDevice->createCPUSideGPUVisibleGPUBufferOnDedMem(creationParams, downloadSize);

				// need a memory dependency to ensure that the compute shader has finished writing to the image
				video::IGPUCommandBuffer::SImageMemoryBarrier imageBarrier = {};
				imageBarrier.barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
				imageBarrier.barrier.dstAccessMask = asset::EAF_TRANSFER_READ_BIT;
				imageBarrier.oldLayout = asset::EIL_GENERAL;
				imageBarrier.newLayout = asset::EIL_GENERAL;
				imageBarrier.srcQueueFamilyIndex = ~0u;
				imageBarrier.dstQueueFamilyIndex = ~0u;
				imageBarrier.image = outImageGPU;
				imageBarrier.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
				imageBarrier.subresourceRange.levelCount = 1u;
				imageBarrier.subresourceRange.layerCount = 1u;
				cmdbuf->pipelineBarrier(asset::EPSF_COMPUTE_SHADER_BIT, asset::EPSF_TRANSFER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, &imageBarrier);

				asset::ICPUImage::SBufferCopy copyRegion = {};
				copyRegion.imageSubresource.aspectMask = video::IGPUImage::EAF_COLOR_BIT;
				copyRegion.imageSubresource.layerCount = 1u;
				copyRegion.imageExtent = outImageGPU->getCreationParameters().extent;
				cmdbuf->copyImageToBuffer(outImageGPU.get(), asset::EIL_GENERAL, downloadBuffer.get(), 1u, &copyRegion);
			}

			cmdbuf->end();

			logicalDevice->resetFences(1u, &fence.get());

			submitInfo = {};
			submitInfo.commandBufferCount = 1u;
			submitInfo.commandBuffers = &cmdbuf.get();
			queues[CommonAPI::InitOutput::EQT_COMPUTE]->submit(1u, &submitInfo, fence.get());

			logicalDevice->blockForFences(1u, &fence.get());

			video::IDriverMemoryAllocation::MappedMemoryRange memoryRange = {};
			memoryRange.memory = downloadBuffer->getBoundMemory();
			memoryRange.length = downloadBuffer->getMemoryReqs().vulkanReqs.size;
			float* mappedGPUData = reinterpret_cast<float*>(logicalDevice->mapMemory(memoryRange));

			// Todo(achal): This needs to change for more complex images
			memcpy(gpuOutput.data(), mappedGPUData, outImageDim[0] * outImageDim[1] * sizeof(float));
			logicalDevice->unmapMemory(downloadBuffer->getBoundMemory());
		}

		// now test (this will have to change for other formats/real images)
		for (uint32_t j = 0u; j < outImageDim[1]; ++j)
		{
			for (uint32_t i = 0u; i < outImageDim[0]; ++i)
			{
				const uint32_t index = j * outImageDim[0] + i;
				if (gpuOutput[index] != cpuOutput[index])
				{
					printf("Failed at (%u, %u)\n", i, j);
					printf("CPU: %f\n", cpuOutput[index]);
					printf("GPU: %f\n", gpuOutput[index]);
					__debugbreak();
				}
			}
		}

		printf("Passed\n");
	}

	void onAppTerminated_impl() override
	{
		logicalDevice->waitIdle();
	}

	void workLoopBody() override
	{

	}

	bool keepRunning() override
	{
		return false;
	}

private:
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
	core::smart_refctd_ptr<video::IGPURenderpass> renderpass = nullptr;
	std::array<nbl::core::smart_refctd_ptr<video::IGPUFramebuffer>, CommonAPI::InitOutput::MaxSwapChainImageCount> fbos;
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

	APP_CONSTRUCTOR(BlitFilterTestApp);
};

NBL_COMMON_API_MAIN(BlitFilterTestApp)

extern "C" {  _declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; }