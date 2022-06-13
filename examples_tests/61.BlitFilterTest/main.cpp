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
using ScaledTriangleKernel = asset::CScaledImageFilterKernel<CTriangleImageFilterKernel>;
using ScaledKaiserKernel = asset::CScaledImageFilterKernel<CKaiserImageFilterKernel<>>;
using ScaledMitchellKernel = asset::CScaledImageFilterKernel<CMitchellImageFilterKernel<>>;
using ScaledMitchellDerivativeKernel = asset::CDerivativeImageFilterKernel<ScaledMitchellKernel>;
using ScaledChannelIndependentKernel = asset::CChannelIndependentImageFilterKernel<ScaledBoxKernel, ScaledMitchellKernel, ScaledKaiserKernel>;

core::smart_refctd_ptr<ICPUImage> createCPUImage(const core::vectorSIMDu32& dims, const asset::IImage::E_TYPE imageType, const asset::E_FORMAT format, const bool fillWithTestData = false)
{
	IImage::SCreationParams imageParams = {};
	imageParams.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(asset::IImage::ECF_MUTABLE_FORMAT_BIT | asset::IImage::ECF_EXTENDED_USAGE_BIT);
	imageParams.type = imageType;
	imageParams.format = format;
	imageParams.extent = { dims[0], dims[1], dims[2] };
	imageParams.mipLevels = 1u;
	imageParams.arrayLayers = 1u;
	imageParams.samples = asset::ICPUImage::ESCF_1_BIT;
	imageParams.usage = asset::IImage::EUF_SAMPLED_BIT;

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

		double dummyVal = 1.0;
		uint8_t* bytePtr = reinterpret_cast<uint8_t*>(image->getBuffer()->getPointer());
		for (uint64_t k = 0u; k < dims[2]; ++k)
		{
			for (uint64_t j = 0u; j < dims[1]; ++j)
			{
				for (uint64_t i = 0; i < dims[0]; ++i)
				{
					const double dummyValToPut = dummyVal++;
					double decodedPixel[4] = { 0 };
					for (uint32_t ch = 0u; ch < asset::getFormatChannelCount(format); ++ch)
						decodedPixel[ch] = dummyValToPut;
						// decodedPixel[ch] = dist(prng);

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
		return video::IGPUImageView::ET_1D_ARRAY;
	case video::IGPUImage::ET_2D:
		return video::IGPUImageView::ET_2D_ARRAY;
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

		// if (false)
		{
			logger->log("Test #1");

			const core::vectorSIMDu32 inImageDim(59u, 1u, 1u);
			const asset::IImage::E_TYPE inImageType = asset::IImage::ET_1D;
			const asset::E_FORMAT inImageFormat = asset::EF_R32G32B32A32_SFLOAT;
			auto inImage = createCPUImage(inImageDim, inImageType, inImageFormat, true);

			const core::vectorSIMDu32 outImageDim(800u, 1u, 1u);
			const IBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic = IBlitUtilities::EAS_NONE_OR_PREMULTIPLIED;

			const core::vectorSIMDf scaleX(0.35f, 1.f, 1.f, 1.f);
			const core::vectorSIMDf scaleY(1.f, 1.f, 1.f, 1.f);
			const core::vectorSIMDf scaleZ(1.f, 1.f, 1.f, 1.f);

			auto kernelX = ScaledMitchellKernel(scaleX, asset::CMitchellImageFilterKernel());
			auto kernelY = ScaledMitchellKernel(scaleY, asset::CMitchellImageFilterKernel());
			auto kernelZ = ScaledMitchellKernel(scaleZ, asset::CMitchellImageFilterKernel());

			using LutDataType = uint16_t;
			blitTest<LutDataType>(std::move(inImage), outImageDim, kernelX, kernelY, kernelZ, alphaSemantic);
		}

		// if (false)
		{
			logger->log("Test #2");

			const char* pathToInputImage = "../../media/colorexr.exr";
			core::smart_refctd_ptr<asset::ICPUImage> inImage = loadImage(pathToInputImage);
			if (!inImage)
				FATAL_LOG("Failed to load the image at path %s\n", pathToInputImage);

			const auto& inExtent = inImage->getCreationParameters().extent;
			const core::vectorSIMDu32 outImageDim(inExtent.width / 3u, inExtent.height / 7u, inExtent.depth);
			const IBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic = IBlitUtilities::EAS_NONE_OR_PREMULTIPLIED;

			const core::vectorSIMDf scaleX(1.f, 1.f, 1.f, 1.f);
			const core::vectorSIMDf scaleY(1.f, 1.f, 1.f, 1.f);
			const core::vectorSIMDf scaleZ(1.f, 1.f, 1.f, 1.f);

			auto kernelX = ScaledMitchellKernel(scaleX, asset::CMitchellImageFilterKernel());
			auto kernelY = ScaledMitchellKernel(scaleY, asset::CMitchellImageFilterKernel());
			auto kernelZ = ScaledMitchellKernel(scaleZ, asset::CMitchellImageFilterKernel());

			using LutDataType = float;
			blitTest<LutDataType>(std::move(inImage), outImageDim, kernelX, kernelY, kernelZ, alphaSemantic);
		}

		// if (false)
		{
			logger->log("Test #3");

			const core::vectorSIMDu32 inImageDim(2u, 3u, 4u);
			// const core::vectorSIMDu32 inImageDim(5u, 4u, 1u);
			// const core::vectorSIMDu32 inImageDim(4u, 1u, 1u);
			const asset::IImage::E_TYPE inImageType = asset::IImage::ET_3D;
			const asset::E_FORMAT inImageFormat = asset::EF_R32G32B32A32_SFLOAT;
			auto inImage = createCPUImage(inImageDim, inImageType, inImageFormat, true);

			const core::vectorSIMDu32 outImageDim(3u, 4u, 2u);
			// const core::vectorSIMDu32 outImageDim(2u, 3u, 1u);
			// const core::vectorSIMDu32 outImageDim(5u, 1u, 1u);
			const IBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic = IBlitUtilities::EAS_NONE_OR_PREMULTIPLIED;

			// const core::vectorSIMDf scaleX(0.35f, 1.f, 1.f, 1.f);
			// const core::vectorSIMDf scaleY(1.f, 9.f/16.f, 1.f, 1.f);
			// const core::vectorSIMDf scaleZ(1.f, 1.f, 1.f, 1.f);
			const core::vectorSIMDf scaleX(1.f, 1.f, 1.f, 1.f);
			const core::vectorSIMDf scaleY(1.f, 1.f, 1.f, 1.f);
			const core::vectorSIMDf scaleZ(1.f, 1.f, 0.5f, 1.f);

			auto kernelX = ScaledMitchellKernel(scaleX, asset::CMitchellImageFilterKernel());
			auto kernelY = ScaledMitchellKernel(scaleY, asset::CMitchellImageFilterKernel());
			auto kernelZ = ScaledMitchellKernel(scaleZ, asset::CMitchellImageFilterKernel());

			using LutDataType = uint16_t;
			blitTest<LutDataType>(std::move(inImage), outImageDim, kernelX, kernelY, kernelZ, alphaSemantic);
		}

		if (false)
		{
			logger->log("Test #4");

			const char* pathToInputImage = "alpha_test_input.exr";
			core::smart_refctd_ptr<asset::ICPUImage> inImage = loadImage(pathToInputImage);
			if (!inImage)
				FATAL_LOG("Failed to load the image at path %s\n", pathToInputImage);

			const auto& inExtent = inImage->getCreationParameters().extent;
			const core::vectorSIMDu32 outImageDim(inExtent.width / 3u, inExtent.height / 7u, inExtent.depth);
			const IBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic = IBlitUtilities::EAS_REFERENCE_OR_COVERAGE;
			const float referenceAlpha = 0.5f;

			const core::vectorSIMDf scaleX(1.f, 1.f, 1.f, 1.f);
			const core::vectorSIMDf scaleY(1.f, 1.f, 1.f, 1.f);
			const core::vectorSIMDf scaleZ(1.f, 1.f, 1.f, 1.f);

			auto kernelX = ScaledMitchellKernel(scaleX, asset::CMitchellImageFilterKernel());
			auto kernelY = ScaledMitchellKernel(scaleY, asset::CMitchellImageFilterKernel());
			auto kernelZ = ScaledMitchellKernel(scaleZ, asset::CMitchellImageFilterKernel());

			using LutDataType = float;
			blitTest<LutDataType>(std::move(inImage), outImageDim, kernelX, kernelY, kernelZ, alphaSemantic, referenceAlpha);
		}

		if (false)
		{
			logger->log("Test #5");
			const core::vectorSIMDu32 inImageDim(257u, 129u, 63u);
			const asset::IImage::E_TYPE inImageType = asset::IImage::ET_3D;
			const asset::E_FORMAT inImageFormat = asset::EF_B10G11R11_UFLOAT_PACK32;
			auto inImage = createCPUImage(inImageDim, inImageType, inImageFormat, true);

			const core::vectorSIMDu32 outImageDim(256u, 128u, 64u);
			const IBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic = IBlitUtilities::EAS_NONE_OR_PREMULTIPLIED;

			const core::vectorSIMDf scaleX(1.f, 1.f, 1.f, 1.f);
			const core::vectorSIMDf scaleY(1.f, 1.f, 1.f, 1.f);
			const core::vectorSIMDf scaleZ(1.f, 1.f, 1.f, 1.f);

			auto kernelX = ScaledMitchellKernel(scaleX, asset::CMitchellImageFilterKernel());
			auto kernelY = ScaledMitchellKernel(scaleY, asset::CMitchellImageFilterKernel());
			auto kernelZ = ScaledMitchellKernel(scaleZ, asset::CMitchellImageFilterKernel());

			using LutDataType = uint16_t;
			blitTest<LutDataType>(std::move(inImage), outImageDim, kernelX, kernelY, kernelZ, alphaSemantic);
		}

		if (false)
		{
			logger->log("Test #6");
			const core::vectorSIMDu32 inImageDim(511u, 1024u, 1u);
			const asset::IImage::E_TYPE inImageType = asset::IImage::ET_2D;
			const asset::E_FORMAT inImageFormat = EF_R16G16B16A16_SNORM;
			auto inImage = createCPUImage(inImageDim, inImageType, inImageFormat, true);

			const core::vectorSIMDu32 outImageDim(512u, 257u, 1u);
			const IBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic = IBlitUtilities::EAS_REFERENCE_OR_COVERAGE;
			const float referenceAlpha = 0.5f;

			const core::vectorSIMDf scaleX(1.f, 1.f, 1.f, 1.f);
			const core::vectorSIMDf scaleY(1.f, 1.f, 1.f, 1.f);
			const core::vectorSIMDf scaleZ(1.f, 1.f, 1.f, 1.f);

			auto kernelX = ScaledMitchellKernel(scaleX, asset::CMitchellImageFilterKernel());
			auto kernelY = ScaledMitchellKernel(scaleY, asset::CMitchellImageFilterKernel());
			auto kernelZ = ScaledMitchellKernel(scaleZ, asset::CMitchellImageFilterKernel());

			using LutDataType = float;
			blitTest<LutDataType>(std::move(inImage), outImageDim, kernelX, kernelY, kernelZ, alphaSemantic, referenceAlpha);
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
	template<typename LutDataType, typename KernelX, typename KernelY, typename KernelZ>
	void blitTest(core::smart_refctd_ptr<asset::ICPUImage>&& inImageCPU, const core::vectorSIMDu32& outExtent, const KernelX& kernelX, const KernelY& kernelY, const KernelZ& kernelZ, const asset::IBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic, const float referenceAlpha = 0.f)
	{
		using BlitFilter = asset::CBlitImageFilter<asset::VoidSwizzle, asset::IdentityDither, void, false, KernelX, KernelY, KernelZ, LutDataType>;

		const asset::E_FORMAT inImageFormat = inImageCPU->getCreationParameters().format;
		const asset::E_FORMAT outImageFormat = inImageFormat; // I can test with different input and output image formats later

		// CPU
		core::vector<uint8_t> cpuOutput(static_cast<uint64_t>(outExtent[0]) * outExtent[1] * outExtent[2] * asset::getTexelOrBlockBytesize(outImageFormat));
		{
			auto outImageCPU = createCPUImage(outExtent, inImageCPU->getCreationParameters().type, outImageFormat);

			KernelX kernelX_(kernelX);
			KernelY kernelY_(kernelY);
			KernelZ kernelZ_(kernelZ);
			typename BlitFilter::state_type blitFilterState(std::move(kernelX_), std::move(kernelY_), std::move(kernelZ_));

			blitFilterState.inOffsetBaseLayer = core::vectorSIMDu32();
			blitFilterState.inExtentLayerCount = core::vectorSIMDu32(0u, 0u, 0u, inImageCPU->getCreationParameters().arrayLayers) + inImageCPU->getMipSize();
			blitFilterState.inImage = inImageCPU.get();
			blitFilterState.outImage = outImageCPU.get();

			blitFilterState.outOffsetBaseLayer = core::vectorSIMDu32();
			const uint32_t outImageLayerCount = 1u;
			blitFilterState.outExtentLayerCount = core::vectorSIMDu32(outExtent[0], outExtent[1], outExtent[2], 1u);

			blitFilterState.axisWraps[0] = asset::ISampler::ETC_CLAMP_TO_EDGE;
			blitFilterState.axisWraps[1] = asset::ISampler::ETC_CLAMP_TO_EDGE;
			blitFilterState.axisWraps[2] = asset::ISampler::ETC_CLAMP_TO_EDGE;
			blitFilterState.borderColor = asset::ISampler::E_TEXTURE_BORDER_COLOR::ETBC_FLOAT_OPAQUE_WHITE;

			blitFilterState.alphaSemantic = alphaSemantic;

			blitFilterState.scratchMemoryByteSize = BlitFilter::getRequiredScratchByteSize(&blitFilterState);
			blitFilterState.scratchMemory = reinterpret_cast<uint8_t*>(_NBL_ALIGNED_MALLOC(blitFilterState.scratchMemoryByteSize, 32));

			using blit_utils_t = asset::CBlitUtilities<KernelX, KernelY, KernelZ>;
			if (!blit_utils_t::template computeScaledKernelPhasedLUT<LutDataType>(blitFilterState.scratchMemory + BlitFilter::getScaledKernelPhasedLUTByteOffset(&blitFilterState), blitFilterState.inExtentLayerCount, blitFilterState.outExtentLayerCount, blitFilterState.inImage->getCreationParameters().type, kernelX, kernelY, kernelZ))
				logger->log("Failed to compute the LUT for blitting\n", system::ILogger::ELL_ERROR);

			logger->log("CPU begin..");
			if (!BlitFilter::execute(core::execution::par_unseq, &blitFilterState))
				logger->log("Failed to blit\n", system::ILogger::ELL_ERROR);
			logger->log("CPU end..");

			if (alphaSemantic == IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
				logger->log("CPU alpha coverage: %f", system::ILogger::ELL_DEBUG, computeAlphaCoverage(referenceAlpha, outImageCPU.get()));

			if (outImageCPU->getCreationParameters().type == asset::IImage::ET_2D)
			{
				const char* writePath = "cpu_out.exr";
				core::smart_refctd_ptr<asset::ICPUImageView> outImageView = nullptr;
				{
					ICPUImageView::SCreationParams viewParams;
					viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
					viewParams.image = outImageCPU;
					viewParams.format = viewParams.image->getCreationParameters().format;
					viewParams.viewType = getImageViewTypeFromImageType_CPU(viewParams.image->getCreationParameters().type);
					viewParams.subresourceRange.baseArrayLayer = 0u;
					viewParams.subresourceRange.layerCount = outImageCPU->getCreationParameters().arrayLayers;
					viewParams.subresourceRange.baseMipLevel = 0u;
					viewParams.subresourceRange.levelCount = outImageCPU->getCreationParameters().mipLevels;

					outImageView = ICPUImageView::create(std::move(viewParams));
				}

				asset::IAssetWriter::SAssetWriteParams wparams(outImageView.get());
				wparams.logger = logger.get();
				if (!assetManager->writeAsset(writePath, wparams))
					logger->log("Failed to write cpu image at path %s\n", system::ILogger::ELL_ERROR, writePath);
			}

			memcpy(cpuOutput.data(), outImageCPU->getBuffer()->getPointer(), cpuOutput.size());

			_NBL_ALIGNED_FREE(blitFilterState.scratchMemory);
		}

		// GPU
		core::vector<uint8_t> gpuOutput(static_cast<uint64_t>(outExtent[0]) * outExtent[1] * outExtent[2] * asset::getTexelOrBlockBytesize(outImageFormat));
		{
			assert(inImageCPU->getCreationParameters().mipLevels == 1);

			auto transitionImageLayout = [this](core::smart_refctd_ptr<video::IGPUImage>&& image, const asset::E_IMAGE_LAYOUT finalLayout)
			{
				core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf = nullptr;
				logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_COMPUTE].get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cmdbuf);

				auto fence = logicalDevice->createFence(video::IGPUFence::ECF_UNSIGNALED);

				video::IGPUCommandBuffer::SImageMemoryBarrier barrier = {};
				barrier.oldLayout = asset::EIL_UNDEFINED;
				barrier.newLayout = asset::EIL_GENERAL;
				barrier.srcQueueFamilyIndex = ~0u;
				barrier.dstQueueFamilyIndex = ~0u;
				barrier.image = image;
				barrier.subresourceRange.aspectMask = video::IGPUImage::EAF_COLOR_BIT;
				barrier.subresourceRange.levelCount = image->getCreationParameters().mipLevels;
				barrier.subresourceRange.layerCount = image->getCreationParameters().arrayLayers;

				cmdbuf->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
				cmdbuf->pipelineBarrier(asset::EPSF_TOP_OF_PIPE_BIT, asset::EPSF_BOTTOM_OF_PIPE_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, &barrier);
				cmdbuf->end();

				video::IGPUQueue::SSubmitInfo submitInfo = {};
				submitInfo.commandBufferCount = 1u;
				submitInfo.commandBuffers = &cmdbuf.get();
				queues[CommonAPI::InitOutput::EQT_COMPUTE]->submit(1u, &submitInfo, fence.get());
				logicalDevice->blockForFences(1u, &fence.get());
			};

			core::smart_refctd_ptr<video::IGPUImage> inImage = nullptr;
			{
				cpu2gpuParams.beginCommandBuffers();
				auto gpuArray = cpu2gpu.getGPUObjectsFromAssets(&inImageCPU, &inImageCPU + 1ull, cpu2gpuParams);
				cpu2gpuParams.waitForCreationToComplete();
				if (!gpuArray || gpuArray->size() < 1ull || (!(*gpuArray)[0]))
					FATAL_LOG("Cannot convert the inpute CPU image to GPU image\n");

				inImage = gpuArray->begin()[0];

				// Do layout transition to SHADER_READ_ONLY_OPTIMAL 
				// I think it might be a good idea to allow the user to change asset::ICPUImage's initialLayout and have the asset converter
				// do the layout transition for them.
				transitionImageLayout(core::smart_refctd_ptr(inImage), asset::EIL_SHADER_READ_ONLY_OPTIMAL);
			}

			core::smart_refctd_ptr<video::IGPUImage> outImage = nullptr;
			{
				video::IGPUImage::SCreationParams creationParams = {};
				creationParams.flags = video::IGPUImage::ECF_MUTABLE_FORMAT_BIT;
				creationParams.type = inImage->getCreationParameters().type;
				creationParams.format = outImageFormat;
				creationParams.extent = { outExtent.x, outExtent.y, outExtent.z };
				creationParams.mipLevels = inImageCPU->getCreationParameters().mipLevels; // Asset converter will make the mip levels 10 for inImage
				creationParams.arrayLayers = inImage->getCreationParameters().arrayLayers;
				creationParams.samples = video::IGPUImage::ESCF_1_BIT;
				creationParams.tiling = video::IGPUImage::ET_OPTIMAL;
				creationParams.usage = static_cast<video::IGPUImage::E_USAGE_FLAGS>(video::IGPUImage::EUF_STORAGE_BIT | video::IGPUImage::EUF_TRANSFER_SRC_BIT);
				outImage = logicalDevice->createDeviceLocalGPUImageOnDedMem(std::move(creationParams));

				transitionImageLayout(core::smart_refctd_ptr(outImage), asset::EIL_GENERAL);
			}

			// Create resources needed to do the blit
			auto blitFilter = core::make_smart_refctd_ptr<video::CComputeBlit>(core::smart_refctd_ptr(logicalDevice));

			const asset::E_FORMAT outImageViewFormat = blitFilter->getOutImageViewFormat(outImageFormat);

			const auto layersToBlit = inImage->getCreationParameters().arrayLayers;
			core::smart_refctd_ptr<video::IGPUImageView> inImageView = nullptr;
			core::smart_refctd_ptr<video::IGPUImageView> outImageView = nullptr;
			{
				video::IGPUImageView::SCreationParams creationParams = {};
				creationParams.image = inImage;
				creationParams.viewType = getImageViewTypeFromImageType_GPU(inImage->getCreationParameters().type);
				creationParams.format = inImage->getCreationParameters().format;
				creationParams.subresourceRange.aspectMask = video::IGPUImage::EAF_COLOR_BIT;
				creationParams.subresourceRange.baseMipLevel = 0;
				creationParams.subresourceRange.levelCount = 1;
				creationParams.subresourceRange.baseArrayLayer = 0;
				creationParams.subresourceRange.layerCount = layersToBlit;

				video::IGPUImageView::SCreationParams outCreationParams = creationParams;
				outCreationParams.image = outImage;
				outCreationParams.format = outImageViewFormat;

				inImageView = logicalDevice->createGPUImageView(std::move(creationParams));
				outImageView = logicalDevice->createGPUImageView(std::move(outCreationParams));
			}

			core::smart_refctd_ptr<video::IGPUImageView> normalizationInImageView = outImageView;
			core::smart_refctd_ptr<video::IGPUImage> normalizationInImage = outImage;
			auto normalizationInFormat = outImageFormat;
			if (alphaSemantic == IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
			{
				normalizationInFormat = video::CComputeBlit::getIntermediateFormat(outImageFormat);

				if (normalizationInFormat != outImageFormat)
				{
					video::IGPUImage::SCreationParams creationParams = outImage->getCreationParameters();
					creationParams.format = normalizationInFormat;
					creationParams.usage = static_cast<video::IGPUImage::E_USAGE_FLAGS>(video::IGPUImage::EUF_STORAGE_BIT | video::IGPUImage::EUF_SAMPLED_BIT);
					normalizationInImage = logicalDevice->createDeviceLocalGPUImageOnDedMem(std::move(creationParams));
					transitionImageLayout(core::smart_refctd_ptr(normalizationInImage), asset::EIL_GENERAL); // First we do the blit which requires storage image so starting layout is GENERAL

					video::IGPUImageView::SCreationParams viewCreationParams = {};
					viewCreationParams.image = normalizationInImage;
					viewCreationParams.viewType = getImageViewTypeFromImageType_GPU(inImage->getCreationParameters().type);
					viewCreationParams.format = normalizationInImage->getCreationParameters().format;
					viewCreationParams.subresourceRange.aspectMask = video::IGPUImage::EAF_COLOR_BIT;
					viewCreationParams.subresourceRange.baseMipLevel = 0;
					viewCreationParams.subresourceRange.levelCount = 1;
					viewCreationParams.subresourceRange.baseArrayLayer = 0;
					viewCreationParams.subresourceRange.layerCount = layersToBlit;

					normalizationInImageView = logicalDevice->createGPUImageView(std::move(viewCreationParams));
				}
			}
			
			const core::vectorSIMDu32 inExtent(inImage->getCreationParameters().extent.width, inImage->getCreationParameters().extent.height, inImage->getCreationParameters().extent.depth, 1);
			const auto inImageType = inImage->getCreationParameters().type;

			// create scratch buffer
			core::smart_refctd_ptr<video::IGPUBuffer> coverageAdjustmentScratchBuffer = nullptr;
			{
				const size_t scratchSize = blitFilter->getCoverageAdjustmentScratchSize(alphaSemantic, video::CComputeBlit::DefaultAlphaBinCount, layersToBlit);
				if (scratchSize > 0)
				{
					video::IGPUBuffer::SCreationParams creationParams = {};
					creationParams.usage = static_cast<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::EUF_TRANSFER_DST_BIT | video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT);

					coverageAdjustmentScratchBuffer = logicalDevice->createDeviceLocalGPUBufferOnDedMem(creationParams, scratchSize);

					asset::SBufferRange<video::IGPUBuffer> bufferRange = {};
					bufferRange.offset = 0ull;
					bufferRange.size = coverageAdjustmentScratchBuffer->getCachedCreationParams().declaredSize;
					bufferRange.buffer = coverageAdjustmentScratchBuffer;

					core::vector<uint32_t> fillValues(scratchSize / sizeof(uint32_t), 0u);
					utilities->updateBufferRangeViaStagingBuffer(queues[CommonAPI::InitOutput::EQT_COMPUTE], bufferRange, fillValues.data());
				}
			}

			// create scaledKernelPhasedLUT and its view
			core::smart_refctd_ptr<video::IGPUBufferView> scaledKernelPhasedLUTView = nullptr;
			{
				using blit_utils_t = asset::CBlitUtilities<KernelX, KernelY, KernelZ>;
				const auto lutSize = blit_utils_t::template getScaledKernelPhasedLUTSize<LutDataType>(inExtent, outExtent, inImageType, kernelX, kernelY, kernelZ);

				uint8_t* lutMemory = reinterpret_cast<uint8_t*>(_NBL_ALIGNED_MALLOC(lutSize, 32));
				if (!blit_utils_t::template computeScaledKernelPhasedLUT<LutDataType>(lutMemory, inExtent, outExtent, inImageType, kernelX, kernelY, kernelZ))
					FATAL_LOG("Failed to compute scaled kernel phased LUT for the GPU case!\n");

				video::IGPUBuffer::SCreationParams creationParams = {};
				creationParams.usage = static_cast<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT | video::IGPUBuffer::EUF_UNIFORM_TEXEL_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT);
				auto scaledKernelPhasedLUT = logicalDevice->createDeviceLocalGPUBufferOnDedMem(creationParams, lutSize);

				// fill it up with data
				asset::SBufferRange<video::IGPUBuffer> bufferRange = {};
				bufferRange.offset = 0ull;
				bufferRange.size = lutSize;
				bufferRange.buffer = scaledKernelPhasedLUT;
				utilities->updateBufferRangeViaStagingBuffer(queues[CommonAPI::InitOutput::EQT_COMPUTE], bufferRange, lutMemory);

				asset::E_FORMAT bufferViewFormat;
				if constexpr (std::is_same_v<LutDataType, uint16_t>)
					bufferViewFormat = asset::EF_R16G16B16A16_SFLOAT;
				else if constexpr (std::is_same_v<LutDataType, float>)
					bufferViewFormat = asset::EF_R32G32B32A32_SFLOAT;
				else
					assert(false);

				scaledKernelPhasedLUTView = logicalDevice->createGPUBufferView(scaledKernelPhasedLUT.get(), bufferViewFormat, 0ull, scaledKernelPhasedLUT->getCachedCreationParams().declaredSize);

				_NBL_ALIGNED_FREE(lutMemory);
			}

			auto blitDSLayout = blitFilter->getDefaultBlitDescriptorSetLayout(alphaSemantic);
			auto kernelWeightsDSLayout = blitFilter->getDefaultKernelWeightsDescriptorSetLayout();
			auto blitPipelineLayout = blitFilter->getDefaultBlitPipelineLayout(alphaSemantic);

			video::IGPUDescriptorSetLayout* blitDSLayouts_raw[] = { blitDSLayout.get(), kernelWeightsDSLayout.get() };
			uint32_t dsCounts[] = { 2, 1 }; 
			auto descriptorPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &blitDSLayout.get(), &blitDSLayout.get() + 1ull, dsCounts);

			core::smart_refctd_ptr<video::IGPUComputePipeline> blitPipeline = nullptr;
			core::smart_refctd_ptr<video::IGPUDescriptorSet> blitDS = nullptr;
			core::smart_refctd_ptr<video::IGPUDescriptorSet> blitWeightsDS = nullptr;

			core::smart_refctd_ptr<video::IGPUComputePipeline> alphaTestPipeline = nullptr;
			core::smart_refctd_ptr<video::IGPUComputePipeline> normalizationPipeline = nullptr;
			core::smart_refctd_ptr<video::IGPUDescriptorSet> normalizationDS = nullptr;

			if (alphaSemantic == IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
			{
				const auto defaultWorkGroupDims = video::CComputeBlit::getDefaultWorkgroupDims(inImageType, layersToBlit);
				auto alphaTestSpecShader = blitFilter->createAlphaTestSpecializedShader(inImage->getCreationParameters().type, defaultWorkGroupDims);
				alphaTestPipeline = logicalDevice->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(blitPipelineLayout), std::move(alphaTestSpecShader));

				auto normalizationSpecShader = blitFilter->createNormalizationSpecializedShader(normalizationInImage->getCreationParameters().type, outImageFormat, outImageViewFormat, defaultWorkGroupDims);
				normalizationPipeline = logicalDevice->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(blitPipelineLayout), std::move(normalizationSpecShader));

				normalizationDS = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), core::smart_refctd_ptr(blitDSLayout));
				blitFilter->updateDescriptorSet(normalizationDS.get(), nullptr, normalizationInImageView, outImageView, coverageAdjustmentScratchBuffer, nullptr);
			}

			core::vectorSIMDu32 outputTexelsPerWG;
			core::smart_refctd_ptr<video::IGPUSpecializedShader> blitSpecShader = nullptr;
			if (alphaSemantic == IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
			{
				blitFilter->getOutputTexelsPerWorkGroup(outputTexelsPerWG, inExtent, outExtent, normalizationInImage->getCreationParameters().format, inImageType, kernelX, kernelY, kernelZ);
				blitSpecShader = blitFilter->createBlitSpecializedShader(
					inImage->getCreationParameters().format,
					normalizationInImage->getCreationParameters().format,
					normalizationInImage->getCreationParameters().format,
					inImageType,
					inExtent,
					outExtent,
					alphaSemantic,
					kernelX, kernelY, kernelZ,
					outputTexelsPerWG,
					512);
			}
			else
			{
				blitFilter->getOutputTexelsPerWorkGroup(outputTexelsPerWG, inExtent, outExtent, normalizationInImage->getCreationParameters().format, inImageType, kernelX, kernelY, kernelZ);
				blitSpecShader = blitFilter->createBlitSpecializedShader(
					inImage->getCreationParameters().format,
					outImageFormat,
					outImageViewFormat,
					inImageType,
					inExtent,
					outExtent,
					alphaSemantic,
					kernelX, kernelY, kernelZ,
					outputTexelsPerWG,
					512);
			}

			blitPipeline = logicalDevice->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(blitPipelineLayout), std::move(blitSpecShader));
			blitDS = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), core::smart_refctd_ptr(blitDSLayout));
			blitWeightsDS = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), core::smart_refctd_ptr(kernelWeightsDSLayout));

			blitFilter->updateDescriptorSet(blitDS.get(), blitWeightsDS.get(), inImageView, normalizationInImageView, coverageAdjustmentScratchBuffer, scaledKernelPhasedLUTView);

			logger->log("GPU begin..");
			blitFilter->blit(
				queues[CommonAPI::InitOutput::EQT_COMPUTE], alphaSemantic,
				blitDS.get(), alphaTestPipeline.get(),
				blitDS.get(), blitWeightsDS.get(), blitPipeline.get(),
				normalizationDS.get(), normalizationPipeline.get(),
				inExtent, inImageType, inImageFormat, normalizationInImage, kernelX, kernelY, kernelZ, outputTexelsPerWG,
				layersToBlit,
				coverageAdjustmentScratchBuffer, referenceAlpha,
				video::CComputeBlit::DefaultAlphaBinCount, 512);
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

				if (alphaSemantic == IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
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
				downloadRegion.imageExtent = outImage->getCreationParameters().extent;

				// Todo(achal): Transition layout to TRANSFER_SRC_OPTIMAL
				cmdbuf->copyImageToBuffer(outImage.get(), asset::EIL_GENERAL, downloadBuffer.get(), 1u, &downloadRegion);

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
		for (uint64_t k = 0u; k < outExtent[2]; ++k)
		{
			for (uint64_t j = 0u; j < outExtent[1]; ++j)
			{
				for (uint64_t i = 0; i < outExtent[0]; ++i)
				{
					const uint64_t pixelIndex = (k * outExtent[1] * outExtent[0]) + (j * outExtent[0]) + i;
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

						if (std::abs(cpuDecodedPixel[ch]-gpuDecodedPixel[ch]) > 1e-5f)
							__debugbreak();
#endif

						sqErr += (cpuDecodedPixel[ch] - gpuDecodedPixel[ch]) * (cpuDecodedPixel[ch] - gpuDecodedPixel[ch]);
					}
				}
			}
		}

		// compute alpha coverage
		const uint64_t totalPixelCount = static_cast<uint64_t>(outExtent[2]) * outExtent[1] * outExtent[0];
		const double RMSE = core::sqrt(sqErr / totalPixelCount);
		logger->log("RMSE: %f\n", system::ILogger::ELL_DEBUG, RMSE);
	}

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
				return nullptr; // it would be weird if the loaded image is already an image view

			inImage = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(asset);

			inImage->addImageUsageFlags(asset::IImage::EUF_SAMPLED_BIT);
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