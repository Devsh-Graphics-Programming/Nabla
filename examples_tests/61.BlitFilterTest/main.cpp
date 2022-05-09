// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"

using namespace nbl;
using namespace nbl::asset;
using namespace nbl::core;
using namespace nbl::video;

#define FATAL_LOG(x, ...) {logger->log(##x, system::ILogger::ELL_ERROR, __VA_ARGS__); exit(-1);}

using ScaledBoxKernel = asset::CScaledImageFilterKernel<CBoxImageFilterKernel>;
using BlitFilter = asset::CBlitImageFilter<asset::VoidSwizzle, asset::IdentityDither, void, false, ScaledBoxKernel, ScaledBoxKernel, ScaledBoxKernel>;

core::smart_refctd_ptr<ICPUImage> createCPUImage(const core::vectorSIMDu32& dims, const asset::IImage::E_TYPE imageType, const asset::E_FORMAT format, const bool fillWithTestData = false)
{
	IImage::SCreationParams imageParams = {};
	imageParams.flags = asset::IImage::ECF_MUTABLE_FORMAT_BIT;
	imageParams.type = imageType;
	imageParams.format = format;
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

	if (fillWithTestData)
	{
		double pixelValueUpperBound = 20.0;
		if (asset::isNormalizedFormat(format))
			pixelValueUpperBound = 1.00000000001;

		std::uniform_real_distribution<double> dist(0.0, pixelValueUpperBound);
		std::mt19937 prng;

		uint8_t* bytePtr = reinterpret_cast<uint8_t*>(image->getBuffer()->getPointer());
		for (uint64_t k = 0u; k < dims[2]; ++k)
		{
			for (uint64_t j = 0u; j < dims[1]; ++j)
			{
				for (uint64_t i = 0; i < dims[0]; ++i)
				{
					double decodedPixel[4] = { 0 };
					for (uint32_t ch = 0u; ch < asset::getFormatChannelCount(format); ++ch)
						decodedPixel[ch] = dist(prng);

					const uint64_t pixelIndex = (k * dims[1] * dims[0]) + (j * dims[0]) + i;
					asset::encodePixelsRuntime(format, bytePtr + pixelIndex * asset::getTexelOrBlockBytesize(format), decodedPixel);
				}
			}
		}
	}

	return image;
}

static inline asset::IImageView<asset::ICPUImage>::E_TYPE getImageViewTypeFromImageType_CPU(const asset::IImage::E_TYPE type)
{
	switch (type)
	{
	case asset::IImage::ET_1D:
		return asset::ICPUImageView::ET_1D;
	case asset::IImage::ET_2D:
		return asset::ICPUImageView::ET_2D;
	case asset::IImage::ET_3D:
		return asset::ICPUImageView::ET_3D;
	default:
		__debugbreak();
		return static_cast<asset::IImageView<asset::ICPUImage>::E_TYPE>(0u);
	}
}

static inline video::IGPUImageView::E_TYPE getImageViewTypeFromImageType_GPU(const video::IGPUImage::E_TYPE type)
{
	switch (type)
	{
	case video::IGPUImage::ET_1D:
		return video::IGPUImageView::ET_1D;
	case video::IGPUImage::ET_2D:
		return video::IGPUImageView::ET_2D;
	case video::IGPUImage::ET_3D:
		return video::IGPUImageView::ET_3D;
	default:
		__debugbreak();
		return static_cast<video::IGPUImageView::E_TYPE>(0u);
	}
}

class BlitFilterTestApp : public ApplicationBase
{
	constexpr static uint32_t SC_IMG_COUNT = 3u;
	constexpr static uint64_t MAX_TIMEOUT = 99999999999999ull;

	using ds_and_pipeline_t = std::pair< core::smart_refctd_ptr<video::IGPUDescriptorSet>, core::smart_refctd_ptr<video::IGPUComputePipeline>>;

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


		{
			logger->log("Test #1");

			const core::vectorSIMDu32 inImageDim(800u, 1u, 1u);
			const asset::IImage::E_TYPE inImageType = asset::IImage::ET_1D;
			const asset::E_FORMAT inImageFormat = asset::EF_R32G32B32A32_SFLOAT;
			auto inImage = createCPUImage(inImageDim, inImageType, inImageFormat, true);

			const core::vectorSIMDu32 outImageDim(59u, 1u, 1u);
			const CBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic = CBlitUtilities::EAS_NONE_OR_PREMULTIPLIED;

			blitTest(inImage, outImageDim, alphaSemantic);
		}

		{
			logger->log("Test #2");

			const char* pathToInputImage = "../../media/colorexr.exr";
			core::smart_refctd_ptr<asset::ICPUImage> inImage = loadImage(pathToInputImage);
			if (!inImage)
				FATAL_LOG("Failed to load the image at path %s\n", pathToInputImage);

			const auto& inExtent = inImage->getCreationParameters().extent;
			const core::vectorSIMDu32 outImageDim(inExtent.width / 3u, inExtent.height / 7u, inExtent.depth);
			const CBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic = CBlitUtilities::EAS_NONE_OR_PREMULTIPLIED;
			blitTest(inImage, outImageDim, alphaSemantic);
		}

		{
			logger->log("Test #3");

			const char* pathToInputImage = "alpha_test_input.exr";
			core::smart_refctd_ptr<asset::ICPUImage> inImage = loadImage(pathToInputImage);
			if (!inImage)
				FATAL_LOG("Failed to load the image at path %s\n", pathToInputImage);

			const auto& inExtent = inImage->getCreationParameters().extent;
			const core::vectorSIMDu32 outImageDim(inExtent.width / 3u, inExtent.height / 7u, inExtent.depth);
			const CBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic = CBlitUtilities::EAS_REFERENCE_OR_COVERAGE;
			const float referenceAlpha = 0.5f;
			blitTest(inImage, outImageDim, alphaSemantic, referenceAlpha);
		}

		{
			logger->log("Test #4");
			const core::vectorSIMDu32 inImageDim(257u, 129u, 63u);
			const asset::IImage::E_TYPE inImageType = asset::IImage::ET_3D;
			const asset::E_FORMAT inImageFormat = asset::EF_B10G11R11_UFLOAT_PACK32;
			auto inImage = createCPUImage(inImageDim, inImageType, inImageFormat, true);

			const core::vectorSIMDu32 outImageDim(256u, 128u, 64u);
			const CBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic = CBlitUtilities::EAS_NONE_OR_PREMULTIPLIED;
			blitTest(inImage, outImageDim, alphaSemantic);
		}

		{
			logger->log("Test #5");
			const core::vectorSIMDu32 inImageDim(511u, 1024u, 1u);
			const asset::IImage::E_TYPE inImageType = asset::IImage::ET_2D;
			const asset::E_FORMAT inImageFormat = EF_R16G16B16A16_SNORM;
			auto inImage = createCPUImage(inImageDim, inImageType, inImageFormat, true);

			const core::vectorSIMDu32 outImageDim(512u, 257u, 1u);
			const CBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic = CBlitUtilities::EAS_REFERENCE_OR_COVERAGE;
			const float referenceAlpha = 0.5f;
			blitTest(inImage, outImageDim, alphaSemantic, referenceAlpha);
		}
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
	void blitTest(core::smart_refctd_ptr<asset::ICPUImage> inImage, const core::vectorSIMDu32& outImageDim, const CBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic, const float referenceAlpha = 0.f)
	{
		const asset::E_FORMAT inImageFormat = inImage->getCreationParameters().format;
		const asset::E_FORMAT outImageFormat = inImageFormat; // I can test with different input and output image formats later
		const uint32_t inChannelCount = asset::getFormatChannelCount(inImageFormat);

		const core::vectorSIMDf scaleX(1.f, 1.f, 1.f, 1.f);
		const core::vectorSIMDf scaleY(1.f, 1.f, 1.f, 1.f);
		const core::vectorSIMDf scaleZ(1.f, 1.f, 1.f, 1.f);

		auto kernelX = ScaledBoxKernel(scaleX, asset::CBoxImageFilterKernel()); // (-1/2, 1/2)
		auto kernelY = ScaledBoxKernel(scaleY, asset::CBoxImageFilterKernel()); // (-1/2, 1/2)
		auto kernelZ = ScaledBoxKernel(scaleZ, asset::CBoxImageFilterKernel()); // (-1/2, 1/2)

		BlitFilter::state_type blitFilterState(std::move(kernelX), std::move(kernelY), std::move(kernelZ));

		blitFilterState.inOffsetBaseLayer = core::vectorSIMDu32();
		blitFilterState.inExtentLayerCount = core::vectorSIMDu32(0u, 0u, 0u, inImage->getCreationParameters().arrayLayers) + inImage->getMipSize();
		blitFilterState.inImage = inImage.get();

		blitFilterState.outOffsetBaseLayer = core::vectorSIMDu32();
		const uint32_t outImageLayerCount = 1u;
		blitFilterState.outExtentLayerCount = core::vectorSIMDu32(outImageDim[0], outImageDim[1], outImageDim[2], 1u);

		blitFilterState.axisWraps[0] = asset::ISampler::ETC_CLAMP_TO_EDGE;
		blitFilterState.axisWraps[1] = asset::ISampler::ETC_CLAMP_TO_EDGE;
		blitFilterState.axisWraps[2] = asset::ISampler::ETC_CLAMP_TO_EDGE;
		blitFilterState.borderColor = asset::ISampler::E_TEXTURE_BORDER_COLOR::ETBC_FLOAT_OPAQUE_WHITE;

		blitFilterState.alphaSemantic = alphaSemantic;

		blitFilterState.scratchMemoryByteSize = BlitFilter::getRequiredScratchByteSize(&blitFilterState);
		blitFilterState.scratchMemory = reinterpret_cast<uint8_t*>(_NBL_ALIGNED_MALLOC(blitFilterState.scratchMemoryByteSize, 32));

		if (!blitFilterState.computePhaseSupportLUT(&blitFilterState))
			logger->log("Failed to compute the LUT for blitting\n", system::ILogger::ELL_ERROR);

		// CPU
		core::vector<uint8_t> cpuOutput(static_cast<uint64_t>(outImageDim[0]) * outImageDim[1] * outImageDim[2] * asset::getTexelOrBlockBytesize(outImageFormat));
		{
			auto outImage = createCPUImage(outImageDim, inImage->getCreationParameters().type, outImageFormat);

			blitFilterState.outImage = outImage.get();

			logger->log("CPU begin..");
			if (!BlitFilter::execute(core::execution::par_unseq, &blitFilterState))
				logger->log("Failed to blit\n", system::ILogger::ELL_ERROR);
			logger->log("CPU end..");

			if (alphaSemantic == CBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
				logger->log("CPU alpha coverage: %f", system::ILogger::ELL_DEBUG, computeAlphaCoverage(referenceAlpha, outImage.get()));

			if (outImage->getCreationParameters().type == asset::IImage::ET_2D)
			{
				const char* writePath = "cpu_out.exr";
				core::smart_refctd_ptr<asset::ICPUImageView> outImageView = nullptr;
				{
					ICPUImageView::SCreationParams viewParams;
					viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
					viewParams.image = outImage;
					viewParams.format = viewParams.image->getCreationParameters().format;
					viewParams.viewType = getImageViewTypeFromImageType_CPU(viewParams.image->getCreationParameters().type);
					viewParams.subresourceRange.baseArrayLayer = 0u;
					viewParams.subresourceRange.layerCount = outImage->getCreationParameters().arrayLayers;
					viewParams.subresourceRange.baseMipLevel = 0u;
					viewParams.subresourceRange.levelCount = outImage->getCreationParameters().mipLevels;

					outImageView = ICPUImageView::create(std::move(viewParams));
				}

				asset::IAssetWriter::SAssetWriteParams wparams(outImageView.get());
				wparams.logger = logger.get();
				if (!assetManager->writeAsset(writePath, wparams))
					logger->log("Failed to write cpu image at path %s\n", system::ILogger::ELL_ERROR, writePath);
			}

			memcpy(cpuOutput.data(), outImage->getBuffer()->getPointer(), cpuOutput.size());
		}

		// GPU
		core::vector<uint8_t> gpuOutput(static_cast<uint64_t>(outImageDim[0]) * outImageDim[1] * outImageDim[2] * asset::getTexelOrBlockBytesize(outImageFormat));
		{
			const core::vectorSIMDu32 inImageDim(inImage->getCreationParameters().extent.width, inImage->getCreationParameters().extent.height, inImage->getCreationParameters().extent.depth);
			const asset::IImage::E_TYPE inImageType = inImage->getCreationParameters().type;

			auto outImage = createCPUImage(outImageDim, inImage->getCreationParameters().type, outImageFormat);

			inImage->addImageUsageFlags(asset::ICPUImage::EUF_SAMPLED_BIT);
			outImage->addImageUsageFlags(asset::ICPUImage::EUF_STORAGE_BIT);

			video::CComputeBlit blitFilter(logicalDevice.get());

			const asset::E_FORMAT outImageViewFormat = blitFilter.getOutImageViewFormat(outImageFormat);
			core::smart_refctd_ptr<video::IGPUImage> inImageGPU = nullptr;
			core::smart_refctd_ptr<video::IGPUImage> outImageGPU = nullptr;
			core::smart_refctd_ptr<video::IGPUImageView> inImageView = nullptr;
			core::smart_refctd_ptr<video::IGPUImageView> outImageView = nullptr;

			if (!getGPUImagesAndTheirViews(inImage, outImage, &inImageGPU, &outImageGPU, &inImageView, &outImageView, outImageViewFormat))
				FATAL_LOG("Failed to convert CPU images to GPU images\n");

			core::smart_refctd_ptr<video::IGPUBuffer> scratchBuffer = nullptr;
			ds_and_pipeline_t alphaTest = { nullptr, nullptr };
			ds_and_pipeline_t normalization = { nullptr, nullptr };

			const size_t paddedSizeAlphaAtomicCounter = core::alignUp(sizeof(uint32_t), logicalDevice->getPhysicalDevice()->getLimits().SSBOAlignment);
			const size_t scratchSize = paddedSizeAlphaAtomicCounter + CComputeBlit::DefaultAlphaBinCount * sizeof(uint32_t); // Todo(achal): Make CComputeBlit report that

			if (alphaSemantic == CBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
			{
				scratchBuffer = createAndClearScratchBuffer(scratchSize);
				getAlphaTestAndNormalizationDSAndPipeline(&blitFilter, &alphaTest, &normalization, inImageView, outImageView, scratchBuffer);
			}

			const core::vectorSIMDu32 inExtent(inImage->getCreationParameters().extent.width, inImage->getCreationParameters().extent.height, inImage->getCreationParameters().extent.depth);
			const core::vectorSIMDu32 outExtent(outImageDim[0], outImageDim[1], outImageDim[2]);
			core::vectorSIMDf scale = static_cast<core::vectorSIMDf>(inExtent).preciseDivision(static_cast<core::vectorSIMDf>(outExtent));

			// Also, cannot use kernelX/Y/Z.getWindowSize() here because they haven't been scaled (yet) for upscaling/downscaling
			const core::vectorSIMDu32 windowDim = static_cast<core::vectorSIMDu32>(core::ceil(scale * core::vectorSIMDf(scaleX.x, scaleY.y, scaleZ.z)));

			const core::vectorSIMDu32 phaseCount = CBlitUtilities::getPhaseCount(inExtent, outExtent, inImage->getCreationParameters().type);
			core::smart_refctd_ptr<video::IGPUBuffer> phaseSupportLUT = nullptr;
			{
				BlitFilter::value_type* lut = reinterpret_cast<BlitFilter::value_type*>(blitFilterState.scratchMemory + BlitFilter::getPhaseSupportLUTByteOffset(&blitFilterState));

				const size_t lutSize = BlitFilter::getRequiredScratchByteSize(&blitFilterState) - BlitFilter::getPhaseSupportLUTByteOffset(&blitFilterState);

				// lut has the LUT in doubles, I want it in floats
				// Todo(achal): Probably need to pack them as half floats?
				core::vector<float> lutInFloats(lutSize / sizeof(BlitFilter::value_type));
				for (uint32_t i = 0u; i < lutInFloats.size(); ++i)
					lutInFloats[i] = static_cast<float>(lut[i]);

				video::IGPUBuffer::SCreationParams uboCreationParams = {};
				uboCreationParams.usage = static_cast<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::EUF_UNIFORM_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT);
				phaseSupportLUT = logicalDevice->createDeviceLocalGPUBufferOnDedMem(uboCreationParams, lutSize);

				// fill it up with data
				asset::SBufferRange<video::IGPUBuffer> bufferRange = {};
				bufferRange.offset = 0ull;
				bufferRange.size = lutInFloats.size() * sizeof(float);
				bufferRange.buffer = phaseSupportLUT;
				utilities->updateBufferRangeViaStagingBuffer(queues[CommonAPI::InitOutput::EQT_COMPUTE], bufferRange, lutInFloats.data());
			}

			const auto& blitDSLayout = blitFilter.getDefaultBlitDSLayout();
			const auto& blitPipelineLayout = blitFilter.getDefaultBlitPipelineLayout();
			const auto& blitShader = blitFilter.createBlitSpecializedShader(inImageGPU->getCreationParameters().format, outImageGPU->getCreationParameters().format, inImageType, inImageDim, outImageDim, alphaSemantic);
			const auto& blitPipeline = blitFilter.createBlitPipeline(core::smart_refctd_ptr(blitShader));

			const uint32_t blitDSCount = 1u;
			auto blitDescriptorPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &blitDSLayout.get(), &blitDSLayout.get() + 1ull, &blitDSCount);

			auto blitDS = logicalDevice->createGPUDescriptorSet(blitDescriptorPool.get(), core::smart_refctd_ptr(blitDSLayout));
			asset::SBufferRange<video::IGPUBuffer> alphaTestCounter = {};
			asset::SBufferRange<video::IGPUBuffer> alphaHistogram = {};
			if (alphaSemantic == CBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
			{
				CComputeBlit::getDefaultAlphaHistogramBufferRange(&alphaHistogram, scratchBuffer, logicalDevice.get());
				CComputeBlit::getDefaultAlphaTestCounterBufferRange(&alphaTestCounter, scratchBuffer, logicalDevice.get());
			}
			CComputeBlit::updateBlitDescriptorSet(logicalDevice.get(), blitDS.get(), inImageView, outImageView, phaseSupportLUT, alphaHistogram);

			logger->log("GPU begin..");
			blitFilter.blit(queues[CommonAPI::InitOutput::EQT_COMPUTE], alphaSemantic, alphaTest.first.get(), alphaTest.second.get(), blitDS.get(), blitPipeline.get(), normalization.first.get(), normalization.second.get(), inImageDim, inImageType, inImageFormat, outImageGPU, referenceAlpha, &alphaTestCounter);
			logger->log("GPU end..");

			if (outImage->getCreationParameters().type == asset::IImage::ET_2D)
			{
				auto outCPUImageView = ext::ScreenShot::createScreenShot(
					logicalDevice.get(),
					queues[CommonAPI::InitOutput::EQT_COMPUTE],
					nullptr,
					outImageView.get(),
					static_cast<asset::E_ACCESS_FLAGS>(0u),
					asset::EIL_GENERAL);

				if (alphaSemantic == CBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
					logger->log("GPU alpha coverage: %f", system::ILogger::ELL_DEBUG, computeAlphaCoverage(referenceAlpha, outCPUImageView->getCreationParameters().image.get()));

				const char* writePath = "gpu_out.exr";
				asset::IAssetWriter::SAssetWriteParams writeParams(outCPUImageView.get());
				if (!assetManager->writeAsset(writePath, writeParams))
					logger->log("Failed to write image at path %s\n", system::ILogger::ELL_ERROR, writePath);
			}

			// download results to check
			{
				core::smart_refctd_ptr<video::IGPUBuffer> downloadBuffer = nullptr;
				const size_t downloadSize = gpuOutput.size();

				video::IGPUBuffer::SCreationParams creationParams = {};
				creationParams.usage = video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
				downloadBuffer = logicalDevice->createCPUSideGPUVisibleGPUBufferOnDedMem(creationParams, downloadSize);

				core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf = nullptr;
				logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_COMPUTE].get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cmdbuf);
				auto fence = logicalDevice->createFence(video::IGPUFence::ECF_UNSIGNALED);

				cmdbuf->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);

				asset::ICPUImage::SBufferCopy downloadRegion = {};
				downloadRegion.imageSubresource.aspectMask = video::IGPUImage::EAF_COLOR_BIT;
				downloadRegion.imageSubresource.layerCount = 1u;
				downloadRegion.imageExtent = outImageGPU->getCreationParameters().extent;

				cmdbuf->copyImageToBuffer(outImageGPU.get(), asset::EIL_GENERAL, downloadBuffer.get(), 1u, &downloadRegion);

				cmdbuf->end();

				video::IGPUQueue::SSubmitInfo submitInfo = {};
				submitInfo.commandBufferCount = 1u;
				submitInfo.commandBuffers = &cmdbuf.get();
				queues[CommonAPI::InitOutput::EQT_COMPUTE]->submit(1u, &submitInfo, fence.get());

				logicalDevice->blockForFences(1u, &fence.get());

				video::IDriverMemoryAllocation::MappedMemoryRange memoryRange = {};
				memoryRange.memory = downloadBuffer->getBoundMemory();
				memoryRange.length = downloadSize;
				uint8_t* mappedGPUData = reinterpret_cast<uint8_t*>(logicalDevice->mapMemory(memoryRange));

				memcpy(gpuOutput.data(), mappedGPUData, gpuOutput.size());
				logicalDevice->unmapMemory(downloadBuffer->getBoundMemory());
			}
		}

		assert(gpuOutput.size() == cpuOutput.size());

		const uint32_t outChannelCount = asset::getFormatChannelCount(outImageFormat);

		double sqErr = 0.0;
		uint8_t* cpuBytePtr = cpuOutput.data();
		uint8_t* gpuBytePtr = gpuOutput.data();
		for (uint64_t k = 0u; k < outImageDim[2]; ++k)
		{
			for (uint64_t j = 0u; j < outImageDim[1]; ++j)
			{
				for (uint64_t i = 0; i < outImageDim[0]; ++i)
				{
					const uint64_t pixelIndex = (k * outImageDim[1] * outImageDim[0]) + (j * outImageDim[0]) + i;
					core::vectorSIMDu32 dummy;

					const void* cpuEncodedPixel = cpuBytePtr + pixelIndex * asset::getTexelOrBlockBytesize(outImageFormat);
					const void* gpuEncodedPixel = gpuBytePtr + pixelIndex * asset::getTexelOrBlockBytesize(outImageFormat);

					double cpuDecodedPixel[4];
					asset::decodePixelsRuntime(outImageFormat, &cpuEncodedPixel, cpuDecodedPixel, dummy.x, dummy.y);

					double gpuDecodedPixel[4];
					asset::decodePixelsRuntime(outImageFormat, &gpuEncodedPixel, gpuDecodedPixel, dummy.x, dummy.y);

					for (uint32_t ch = 0u; ch < outChannelCount; ++ch)
					{
#if 0
						if (std::isnan(cpuDecodedPixel[ch]) || std::isinf(cpuDecodedPixel[ch]))
							__debugbreak();

						if (std::isnan(gpuDecodedPixel[ch]) || std::isinf(gpuDecodedPixel[ch]))
							__debugbreak();

						if (cpuDecodedPixel[ch] != gpuDecodedPixel[ch])
							__debugbreak();
#endif

						sqErr += (cpuDecodedPixel[ch] - gpuDecodedPixel[ch]) * (cpuDecodedPixel[ch] - gpuDecodedPixel[ch]);
					}
				}
			}
		}

		// compute alpha coverage
		const uint64_t totalPixelCount = static_cast<uint64_t>(outImageDim[2]) * outImageDim[1] * outImageDim[0];
		const double RMSE = core::sqrt(sqErr / totalPixelCount);
		logger->log("RMSE: %f\n", system::ILogger::ELL_DEBUG, RMSE);

		_NBL_ALIGNED_FREE(blitFilterState.scratchMemory);
	}

	core::smart_refctd_ptr<video::IGPUBuffer> createAndClearScratchBuffer(const size_t scratchSize)
	{
		video::IGPUBuffer::SCreationParams creationParams = {};
		creationParams.usage = static_cast<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::EUF_TRANSFER_DST_BIT | video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT);

		auto result = logicalDevice->createDeviceLocalGPUBufferOnDedMem(creationParams, scratchSize);

		asset::SBufferRange<video::IGPUBuffer> bufferRange = {};
		bufferRange.offset = 0ull;
		bufferRange.size = result->getCachedCreationParams().declaredSize;
		bufferRange.buffer = result;

		core::vector<uint32_t> fillValues(scratchSize / sizeof(uint32_t), 0u);
		utilities->updateBufferRangeViaStagingBuffer(queues[CommonAPI::InitOutput::EQT_COMPUTE], bufferRange, fillValues.data());

		return result;
	};

	void getAlphaTestAndNormalizationDSAndPipeline(video::CComputeBlit* blitFilter, ds_and_pipeline_t* alphaTest, ds_and_pipeline_t* normalization, core::smart_refctd_ptr<video::IGPUImageView> inImageView, core::smart_refctd_ptr<video::IGPUImageView> outImageView, core::smart_refctd_ptr<video::IGPUBuffer> scratchBuffer)
	{
		const asset::IImage::E_TYPE inImageType = inImageView->getCreationParameters().image->getCreationParameters().type;

		auto& alphaTestDS = alphaTest->first;
		auto& alphaTestPipeline = alphaTest->second;

		const auto& alphaTestDSLayout = blitFilter->getDefaultAlphaTestDSLayout();
		const auto& alphaTestCompShader = blitFilter->createAlphaTestSpecializedShader(inImageType);
		const auto& alphaTestPipelineLayout = blitFilter->getDefaultAlphaTestPipelineLayout();
		alphaTestPipeline = blitFilter->createAlphaTestPipeline(core::smart_refctd_ptr(alphaTestCompShader));

		const uint32_t alphaTestDSCount = 1u;
		auto alphaTestDescriptorPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &alphaTestDSLayout.get(), &alphaTestDSLayout.get() + 1ull, &alphaTestDSCount);

		alphaTestDS = logicalDevice->createGPUDescriptorSet(alphaTestDescriptorPool.get(), core::smart_refctd_ptr(alphaTestDSLayout));

		asset::SBufferRange<video::IGPUBuffer> alphaTestCounter = {};
		CComputeBlit::getDefaultAlphaTestCounterBufferRange(&alphaTestCounter, scratchBuffer, logicalDevice.get());
		CComputeBlit::updateAlphaTestDescriptorSet(logicalDevice.get(), alphaTestDS.get(), inImageView, alphaTestCounter);

		auto& normalizationDS = normalization->first;
		auto& normalizationPipeline = normalization->second;

		const auto& normalizationDSLayout = blitFilter->getDefaultNormalizationDSLayout();
		const auto& normalizationCompShader = blitFilter->createNormalizationSpecializedShader(inImageType, outImageView->getCreationParameters().image->getCreationParameters().format); // Todo(achal): Rethink format param, view or image?
		const auto& normalizationPipelineLayout = blitFilter->getDefaultNormalizationPipelineLayout();
		normalizationPipeline = blitFilter->createNormalizationPipeline(core::smart_refctd_ptr(normalizationCompShader));

		const uint32_t normalizationDSCount = 1u;
		auto normalizationDescriptorPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &normalizationDSLayout.get(), &normalizationDSLayout.get() + 1ull, &normalizationDSCount);

		normalizationDS = logicalDevice->createGPUDescriptorSet(normalizationDescriptorPool.get(), core::smart_refctd_ptr(normalizationDSLayout));

		asset::SBufferRange<video::IGPUBuffer> alphaHistogram = {};
		CComputeBlit::getDefaultAlphaHistogramBufferRange(&alphaHistogram, scratchBuffer, logicalDevice.get());
		CComputeBlit::updateNormalizationDescriptorSet(logicalDevice.get(), normalizationDS.get(), outImageView, alphaHistogram, alphaTestCounter);
	};

	float computeAlphaCoverage(const double referenceAlpha, asset::ICPUImage* image)
	{
		const uint32_t mipLevel = 0u;

		uint32_t alphaTestPassCount = 0u;
		const auto& extent = image->getCreationParameters().extent;

		for (uint32_t z = 0u; z < extent.depth; ++z)
		{
			for (uint32_t y = 0u; y < extent.height; ++y)
			{
				for (uint32_t x = 0u; x < extent.width; ++x)
				{
					const core::vectorSIMDu32 texCoord(x, y, z);
					core::vectorSIMDu32 dummy;
					const void* encodedPixel = image->getTexelBlockData(mipLevel, texCoord, dummy);

					double decodedPixel[4];
					asset::decodePixelsRuntime(image->getCreationParameters().format, &encodedPixel, decodedPixel, dummy.x, dummy.y);

					if (decodedPixel[3] > referenceAlpha)
						++alphaTestPassCount;
				}
			}
		}

		const float alphaCoverage = float(alphaTestPassCount) / float(extent.width * extent.height * extent.depth);
		return alphaCoverage;
	};

	bool getGPUImagesAndTheirViews(
		core::smart_refctd_ptr<asset::ICPUImage> inCPU,
		core::smart_refctd_ptr<asset::ICPUImage> outCPU,
		core::smart_refctd_ptr<video::IGPUImage>* inGPU,
		core::smart_refctd_ptr<video::IGPUImage>* outGPU,
		core::smart_refctd_ptr<video::IGPUImageView>* inGPUView,
		core::smart_refctd_ptr<video::IGPUImageView>* outGPUView,
		const asset::E_FORMAT outGPUViewFormat)
	{
		core::smart_refctd_ptr<asset::ICPUImage> tmp[2] = { inCPU, outCPU };
		cpu2gpuParams.beginCommandBuffers();
		auto gpuArray = cpu2gpu.getGPUObjectsFromAssets(tmp, tmp + 2ull, cpu2gpuParams);
		cpu2gpuParams.waitForCreationToComplete();
		if (!gpuArray || gpuArray->size() < 2ull || (!(*gpuArray)[0]))
			return false;

		*inGPU = gpuArray->begin()[0];
		*outGPU = gpuArray->begin()[1];

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
		barriers[0].image = *inGPU;
		barriers[0].subresourceRange.aspectMask = video::IGPUImage::EAF_COLOR_BIT;
		barriers[0].subresourceRange.levelCount = 1u;
		barriers[0].subresourceRange.layerCount = 1u;

		barriers[1] = barriers[0];
		barriers[1].image = *outGPU;

		cmdbuf->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
		cmdbuf->pipelineBarrier(asset::EPSF_TOP_OF_PIPE_BIT, asset::EPSF_BOTTOM_OF_PIPE_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 2u, barriers);
		cmdbuf->end();

		video::IGPUQueue::SSubmitInfo submitInfo = {};
		submitInfo.commandBufferCount = 1u;
		submitInfo.commandBuffers = &cmdbuf.get();
		queues[CommonAPI::InitOutput::EQT_COMPUTE]->submit(1u, &submitInfo, fence.get());
		logicalDevice->blockForFences(1u, &fence.get());

		// create views for images
		{
			video::IGPUImageView::SCreationParams inCreationParams = {};
			inCreationParams.image = *inGPU;
			inCreationParams.viewType = getImageViewTypeFromImageType_GPU((*inGPU)->getCreationParameters().type);
			inCreationParams.format = (*inGPU)->getCreationParameters().format;
			inCreationParams.subresourceRange.aspectMask = video::IGPUImage::EAF_COLOR_BIT;
			inCreationParams.subresourceRange.layerCount = 1u;
			inCreationParams.subresourceRange.levelCount = 1u;

			video::IGPUImageView::SCreationParams outCreationParams = inCreationParams;
			outCreationParams.image = *outGPU;
			outCreationParams.format = outGPUViewFormat;

			*inGPUView = logicalDevice->createGPUImageView(std::move(inCreationParams));
			*outGPUView = logicalDevice->createGPUImageView(std::move(outCreationParams));
		}

		return true;
	};

	core::smart_refctd_ptr<asset::ICPUImage> loadImage(const char* path)
	{
		core::smart_refctd_ptr<asset::ICPUImage> inImage = nullptr;
		{
			constexpr auto cachingFlags = static_cast<nbl::asset::IAssetLoader::E_CACHING_FLAGS>(nbl::asset::IAssetLoader::ECF_DONT_CACHE_REFERENCES & nbl::asset::IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL);

			asset::IAssetLoader::SAssetLoadParams loadParams(0ull, nullptr, cachingFlags);
			auto cpuImageBundle = assetManager->getAsset(path, loadParams);
			auto cpuImageContents = cpuImageBundle.getContents();
			if (cpuImageContents.empty() || cpuImageContents.begin() == cpuImageContents.end())
				return nullptr;

			auto asset = *cpuImageContents.begin();
			if (asset->getAssetType() == asset::IAsset::ET_IMAGE_VIEW)
				__debugbreak(); // it would be weird if the loaded image is already an image view

			inImage = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(asset);
		}
		return inImage;
	};


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