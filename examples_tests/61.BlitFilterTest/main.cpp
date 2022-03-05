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

core::smart_refctd_ptr<ICPUImage> createCPUImage(const std::array<uint32_t, 3>& dims, const asset::IImage::E_TYPE imageType)
{
	IImage::SCreationParams imageParams = {};
	imageParams.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);
	imageParams.type = imageType;
	imageParams.format = asset::EF_R32_SFLOAT;
	imageParams.extent = { dims[0], dims[1], dims[2] };
	imageParams.mipLevels = 1u;
	imageParams.arrayLayers = 1u;
	imageParams.samples = asset::ICPUImage::ESCF_1_BIT;

	auto imageRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::IImage::SBufferCopy>>(1ull);
	auto& region = (*imageRegions)[0];
	region.bufferImageHeight = 0u;
	region.bufferOffset = 0ull;
	region.bufferRowLength = dims[0];
	region.imageExtent = { dims[0], dims[1], dims[2] };
	region.imageOffset = { 0u, 0u, 0u };
	region.imageSubresource.baseArrayLayer = 0u;
	region.imageSubresource.layerCount = 1u;
	region.imageSubresource.mipLevel = 0;

	size_t bufferSize = asset::getTexelOrBlockBytesize(imageParams.format) * static_cast<size_t>(region.imageExtent.width) * region.imageExtent.height * region.imageExtent.depth;
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

		std::array<uint32_t, 3> inImageDim = { 1023u, 1u, 1u }; // { 800u, 5u };
		if (inImageDim[0]*inImageDim[1]*inImageDim[2] > 1024) // until this holds I will NOT run of of smem as well, given my current format
			__debugbreak();
		std::array<uint32_t, 3> outImageDim = { 1u, 1u, 1u }; // { 16u, 69u };

		core::smart_refctd_ptr<asset::ICPUImage> inImage = createCPUImage(inImageDim, asset::IImage::ET_3D);

		float inputVal = 1.f;
		float* inImagePixel = reinterpret_cast<float*>(inImage->getBuffer()->getPointer());
		for (uint32_t k = 0u; k < inImageDim[2]; ++k)
		{
			for (uint32_t j = 0u; j < inImageDim[1]; ++j)
			{
				for (uint32_t i = 0; i < inImageDim[0]; ++i)
				{
					inImagePixel[k * (inImageDim[1]*inImageDim[0]) + j * inImageDim[0] + i] = inputVal++;
				}
			}
		}
		
		const core::vectorSIMDf scaleX(1.f, 1.f, 1.f, 1.f);
		const core::vectorSIMDf scaleY(1.f, 1.f, 1.f, 1.f);
		const core::vectorSIMDf scaleZ(1.f, 1.f, 1.f, 1.f);

		// CPU blit
		core::vector<float> cpuOutput(inImageDim[0] * inImageDim[1] * inImageDim[2]);
#if 0
		{
			core::smart_refctd_ptr<ICPUImage> outImage = createCPUImage(outImageDim, asset::IImage::ET_3D);

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

			// This needs to change when testing with more complex images
			float* outPixel = reinterpret_cast<float*>(outImage->getBuffer()->getPointer());
			// memcpy(cpuOutput.data(), outPixel, cpuOutput.size()*sizeof(float));
		}

		for (uint32_t k = 0u; k < outImageDim[2]; ++k)
		{
			for (uint32_t j = 0u; j < outImageDim[1]; ++j)
			{
				for (uint32_t i = 0u; i < outImageDim[0]; ++i)
				{
					const uint32_t index = k * (outImageDim[1] * outImageDim[0]) + j * outImageDim[0] + i;
					printf("%f\t", cpuOutput[index]);
				}
				printf("\n");
			}

			printf("\n\n");
		}
#endif

		// GPU blit
		core::vector<float> gpuOutput(static_cast<uint64_t>(outImageDim[0]) * outImageDim[1] * outImageDim[2]);
		{
			// it is probably a good idea to expose VkPhysicalDeviceLimits::maxComputeSharedMemorySize
			constexpr size_t MAX_SMEM_SIZE = 16ull * 1024ull;

			const core::vectorSIMDf inExtent(inImageDim[0], inImageDim[1], inImageDim[2]);
			const core::vectorSIMDf outExtent(outImageDim[0], outImageDim[1], outImageDim[2]);

			const core::vectorSIMDf scale = inExtent.preciseDivision(outExtent);
			
			// kernelX/Y/Z stores absolute values of support so they won't be helpful here
			const core::vectorSIMDf negativeSupport = core::vectorSIMDf(-0.5f, -0.5f, -0.5f)*scale;
			const core::vectorSIMDf positiveSupport = core::vectorSIMDf(0.5f, 0.5f, 0.5f)*scale;

			// I think this formulation is better than ceil(scaledPositiveSupport-scaledNegativeSupport) because if scaledPositiveSupport comes out a tad bit
			// greater than what it should be and if scaledNegativeSupport comes out a tad bit smaller than it should be then the distance between them would
			// become a tad bit greater than it should be and when we take a ceil it'll jump up to the next integer thus giving us 1 more than the actual window
			// size, example: 49x1 -> 7x1.
			// Also, cannot use kernelX/Y/Z.getWindowSize() here because they haven't been scaled (yet) for upscaling/downscaling
			const core::vectorSIMDu32 windowDim = static_cast<core::vectorSIMDu32>(core::ceil(scale*core::vectorSIMDf(scaleX.x, scaleY.y, scaleZ.z)));

			// fail if the window cannot be preloaded into shared memory
			const size_t windowSize = static_cast<size_t>(windowDim.x) * windowDim.y * windowDim.z * asset::getTexelOrBlockBytesize(inImage->getCreationParameters().format);
			if (windowSize > MAX_SMEM_SIZE)
			{
				printf("Failed to blit because supports are too large\n");
				__debugbreak();
			}

			const auto& limits = physicalDevice->getLimits();
			constexpr uint32_t MAX_INVOCATION_COUNT = 1024u;
			// matching workgroup size DIRECTLY to windowDims for now --this does mean that unfortunately my workgroups will be sized weirdly, and sometimes
			// they could be too small as well
			const uint32_t totalInvocationCount = windowDim.x * windowDim.y * windowDim.z;
			if ((totalInvocationCount > MAX_INVOCATION_COUNT) || (windowDim.x > limits.maxWorkgroupSize[0]) || (windowDim.y > limits.maxWorkgroupSize[1]) || (windowDim.z > limits.maxWorkgroupSize[2]))
			{
				printf("Failed to blit because workgroup size limit exceeded\n");
				__debugbreak();
			}

			core::smart_refctd_ptr<ICPUImage> outImage = createCPUImage(outImageDim, asset::IImage::ET_3D);

#if 1
			struct alignas(16) vec3_aligned
			{
				float x, y, z;
			};

			struct alignas(16) uvec3_aligned
			{
				uint32_t x, y, z;
			};
#endif

			struct push_constants_t
			{
				uvec3_aligned inDim;
				uvec3_aligned outDim;
				vec3_aligned negativeSupport;
				vec3_aligned positiveSupport;
				uvec3_aligned windowDim;

#if 0
				uint32_t inWidth;
				uint32_t outWidth;

				float negativeSupport;
				float positiveSupport;

				uint32_t windowDim;
#endif
			};

#if 0
			core::smart_refctd_ptr<video::IGPUBuffer> phaseSupportLUT = nullptr;
			{
				const core::vectorSIMDu32 inExtent(inImage->getCreationParameters().extent.width, inImage->getCreationParameters().extent.height, inImage->getCreationParameters().extent.depth);
				const core::vectorSIMDu32 outExtent(outImage->getCreationParameters().extent.width, outImage->getCreationParameters().extent.height, outImage->getCreationParameters().extent.depth);
				const core::vectorSIMDu32 phaseCount = BlitFilter::getPhaseCount(inExtent, outExtent, inImage->getCreationParameters().type);

				// create a blit filter state just compute the LUT (this will change depending on how we choose to expose this to the user i.e. do we want
				// the same API as asset::CBlitImageFilter or something else?)

				auto kernelX = ScaledBoxKernel(scaleX, asset::CBoxImageFilterKernel()); // (-1/2, 1/2)
				auto kernelY = ScaledBoxKernel(scaleY, asset::CBoxImageFilterKernel()); // (-1/2, 1/2)
				auto kernelZ = ScaledBoxKernel(scaleZ, asset::CBoxImageFilterKernel()); // (-1/2, 1/2)
				BlitFilter::state_type blitFilterState(std::move(kernelX), std::move(kernelY), std::move(kernelZ));

				blitFilterState.inOffsetBaseLayer = core::vectorSIMDu32();
				blitFilterState.inExtentLayerCount = inExtent + inImage->getMipSize();
				blitFilterState.inImage = inImage.get();

				blitFilterState.outOffsetBaseLayer = core::vectorSIMDu32();
				blitFilterState.outExtentLayerCount = outExtent + outImage->getMipSize();
				blitFilterState.outImage = outImage.get();

				blitFilterState.axisWraps[0] = asset::ISampler::ETC_CLAMP_TO_EDGE;
				blitFilterState.axisWraps[1] = asset::ISampler::ETC_CLAMP_TO_EDGE;
				blitFilterState.axisWraps[2] = asset::ISampler::ETC_CLAMP_TO_EDGE;
				blitFilterState.borderColor = asset::ISampler::E_TEXTURE_BORDER_COLOR::ETBC_FLOAT_OPAQUE_WHITE;

				blitFilterState.scratchMemoryByteSize = BlitFilter::getRequiredScratchByteSize(&blitFilterState);
				blitFilterState.scratchMemory = reinterpret_cast<uint8_t*>(_NBL_ALIGNED_MALLOC(blitFilterState.scratchMemoryByteSize, 32));

				blitFilterState.computePhaseSupportLUT(&blitFilterState);
				BlitFilter::value_type* lut = reinterpret_cast<BlitFilter::value_type*>(blitFilterState.scratchMemory + BlitFilter::getPhaseSupportLUTByteOffset(&blitFilterState));

				// need to compute the size
				// take each element of the LUT and convert to half floats and pack it

				// const size_t neededSize = ;
				// video::IGPUBuffer::SCreationParams uboCreationParams = {};
				// uboCreationParams.usage = static_cast<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::EUF_UNIFORM_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT);
				// phaseSupportLUT = logicalDevice->createDeviceLocalGPUBufferOnDedMem(uboCreationParams, neededSize);
			}
#endif

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
			if (!gpuArray || gpuArray->size() < 2ull || (!(*gpuArray)[0]))
				FATAL_LOG("Failed to convert CPU images to GPU images\n");

			inImageGPU = gpuArray->begin()[0];
			outImageGPU = gpuArray->begin()[1];

			// do layout transition to GENERAL
			// (I think it might be a good idea to allow the user to change asset::ICPUImage's initialLayout and have the asset converter
			// do the layout transition for them)
			core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf = nullptr;
			logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_COMPUTE].get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cmdbuf);
			auto fence = logicalDevice->createFence(video::IGPUFence::ECF_UNSIGNALED);
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

			video::IGPUQueue::SSubmitInfo submitInfo = {};
			submitInfo.commandBufferCount = 1u;
			submitInfo.commandBuffers = &cmdbuf.get();
			queues[CommonAPI::InitOutput::EQT_COMPUTE]->submit(1u, &submitInfo, fence.get());
			logicalDevice->blockForFences(1u, &fence.get());

			// create image views for images
			{
				video::IGPUImageView::SCreationParams inCreationParams = {};
				inCreationParams.image = inImageGPU;
				inCreationParams.viewType = video::IGPUImageView::ET_3D;
				inCreationParams.format = inImageGPU->getCreationParameters().format;
				inCreationParams.subresourceRange.aspectMask = video::IGPUImage::EAF_COLOR_BIT;
				inCreationParams.subresourceRange.layerCount = 1u;
				inCreationParams.subresourceRange.levelCount = 1u;

				video::IGPUImageView::SCreationParams outCreationParams = inCreationParams;
				outCreationParams.image = outImageGPU;
				outCreationParams.format = outImageGPU->getCreationParameters().format;

				inImageView = logicalDevice->createGPUImageView(std::move(inCreationParams));
				outImageView = logicalDevice->createGPUImageView(std::move(outCreationParams));
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

			asset::SPushConstantRange pcRange = {};
			pcRange.stageFlags = asset::IShader::ESS_COMPUTE;
			pcRange.offset = 0u;
			pcRange.size = sizeof(push_constants_t);
			core::smart_refctd_ptr<video::IGPUPipelineLayout> pipelineLayout = logicalDevice->createGPUPipelineLayout(&pcRange, &pcRange + 1ull, core::smart_refctd_ptr(dsLayout));

			const uint32_t dsCount = 1u;
			auto descriptorPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &dsLayout.get(), &dsLayout.get() + 1ull, &dsCount);

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

				auto unspecOverridenShader = asset::IGLSLCompiler::createOverridenCopy(loadedShader->getUnspecialized(),
					"#define _NBL_GLSL_WORKGROUP_SIZE_X_ %d\n"
					"#define _NBL_GLSL_WORKGROUP_SIZE_Y_ %d\n"
					"#define _NBL_GLSL_WORKGROUP_SIZE_Z_ %d\n",
					windowDim.x, windowDim.y, windowDim.z);

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
			push_constants_t pc = { };
			{
				pc.inDim.x = inImageDim[0]; pc.inDim.y = inImageDim[1]; pc.inDim.z = inImageDim[2];
				pc.outDim.x = outImageDim[0]; pc.outDim.y = outImageDim[1]; pc.outDim.z = outImageDim[2];
				pc.negativeSupport.x = negativeSupport.x; pc.negativeSupport.y = negativeSupport.y; pc.negativeSupport.z = negativeSupport.z;
				pc.positiveSupport.x = positiveSupport.x; pc.positiveSupport.y = positiveSupport.y; pc.positiveSupport.z = positiveSupport.z;
				pc.windowDim.x = windowDim.x; pc.windowDim.y = windowDim.y; pc.windowDim.z = windowDim.z;
			}
			cmdbuf->pushConstants(pipeline->getLayout(), asset::IShader::ESS_COMPUTE, 0u, sizeof(push_constants_t), &pc);

			const uint32_t totalWindowCount = outImageDim[0];
			// const uint32_t windowDim = std::ceil(inImageDim[0] / outImageDim[0]);
			// const uint32_t windowsPerWG = WG_DIM / windowDim; // we want to make sure that the WG covers a window COMPLETELY even if it means throwing away some invocations
			// const uint32_t wgCount = (totalWindowCount + windowsPerWG - 1) / windowsPerWG;

			cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, pipeline->getLayout(), 0u, 1u, &ds.get());
			// cmdbuf->dispatch(wgCount, 1u, 1u);
			cmdbuf->dispatch(1u, 1u, 1u);

			// after the dispatch download the buffer to check, one buffer to download contents from all images
			core::smart_refctd_ptr<video::IGPUBuffer> downloadBuffer = nullptr;
			const size_t downloadSize = static_cast<size_t>(outImageDim[0]) * outImageDim[1] * outImageDim[2] * asset::getTexelOrBlockBytesize(outImageGPU->getCreationParameters().format);
			{
				video::IGPUBuffer::SCreationParams creationParams = {};
				creationParams.usage = video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
				downloadBuffer = logicalDevice->createCPUSideGPUVisibleGPUBufferOnDedMem(creationParams, downloadSize);

				// need a memory dependency to ensure that the compute shader has finished writing to the image(s)
				asset::SMemoryBarrier memoryBarrier = { asset::EAF_SHADER_WRITE_BIT, asset::EAF_TRANSFER_READ_BIT };
				video::IGPUCommandBuffer::SImageMemoryBarrier imageBarrier = {};
				cmdbuf->pipelineBarrier(asset::EPSF_COMPUTE_SHADER_BIT, asset::EPSF_TRANSFER_BIT, asset::EDF_NONE, 1u, &memoryBarrier, 0u, nullptr, 0u, nullptr);
			}

			asset::ICPUImage::SBufferCopy copyRegion = {};
			copyRegion.imageSubresource.aspectMask = video::IGPUImage::EAF_COLOR_BIT;
			copyRegion.imageSubresource.layerCount = 1u;
			copyRegion.imageExtent = outImageGPU->getCreationParameters().extent;

			cmdbuf->copyImageToBuffer(outImageGPU.get(), asset::EIL_GENERAL, downloadBuffer.get(), 1u, &copyRegion);
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
			memcpy(gpuOutput.data(), mappedGPUData, static_cast<size_t>(outImageDim[0]) * outImageDim[1] * outImageDim[2] * sizeof(float));
			logicalDevice->unmapMemory(downloadBuffer->getBoundMemory());
		}

		// now test (this will have to change for other formats/real images)
		for (uint32_t k = 0u; k < outImageDim[2]; ++k)
		{
			for (uint32_t j = 0u; j < outImageDim[1]; ++j)
			{
				for (uint32_t i = 0u; i < outImageDim[0]; ++i)
				{
					const uint32_t index = k * (outImageDim[1] * outImageDim[0]) + j * outImageDim[0] + i;
					if (gpuOutput[index] != cpuOutput[index])
					{
						printf("Failed at (%u, %u, %u)\n", i, j, k);
						printf("CPU: %f\n", cpuOutput[index]);
						printf("GPU: %f\n", gpuOutput[index]);
						__debugbreak();
					}
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