// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "../common/Camera.hpp"
#include "../common/CommonAPI.h"

using namespace nbl;
using namespace asset;
using namespace core;
using namespace video;
using namespace ui;

using SATFilter = CSummedAreaTableImageFilter<false>;

static core::smart_refctd_ptr<ICPUBuffer> computeLuminancePdf(smart_refctd_ptr<ICPUImage> envmap, float* normalizationFactor)
{
	const core::vector2d<uint32_t> envmapExtent = { envmap->getCreationParameters().extent.width, envmap->getCreationParameters().extent.height };
	const uint32_t channelCount = getFormatChannelCount(envmap->getCreationParameters().format);
	
	const core::vector2d<uint32_t> pdfDomainExtent = { envmapExtent.X, envmapExtent.Y };

	const size_t outBufferSize = pdfDomainExtent.X * pdfDomainExtent.Y * sizeof(double);
	core::smart_refctd_ptr<ICPUBuffer> outBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(outBufferSize);

	const double luminanceScales[4] = { 0.2126729 , 0.7151522, 0.0721750, 0.0 };

	float* envmapPixel = (float*)envmap->getBuffer()->getPointer();
	double* outPixel = (double*)outBuffer->getPointer();

	double pdfSum = 0.0;

	for (uint32_t y = 0; y < pdfDomainExtent.Y; ++y)
	{
		const double sinTheta = core::sin(core::PI<double>() * ((y + 0.5) / (double)pdfDomainExtent.Y));

		for (uint32_t x = 0; x < pdfDomainExtent.X; ++x)
		{
			double result = 0.0;
			for (uint32_t ch = 0; ch < channelCount; ++ch)
				result += luminanceScales[ch] * envmapPixel[ch];

			*outPixel++ = result * sinTheta;
			pdfSum += result * sinTheta;
			envmapPixel += channelCount;
		}
	}

	if (normalizationFactor)
		*normalizationFactor = (pdfDomainExtent.X * pdfDomainExtent.Y)/(pdfSum*2.0*core::PI<double>()*core::PI<double>());

	return outBuffer;
}

// Returns the offset into the passed array the element at which is <= the passed element (`x`)
// returns offset = -1 if passed element is < the element at index 0
static int32_t bisectionSearch(const double* arr, const uint32_t arrCount, const double x, double* xFound)
{
	int32_t offset = std::upper_bound(arr, arr + arrCount, x) - arr - 1u;
	double dx = 0.0;
	if (offset == -1)
		dx = x / arr[offset + 1];
	else
		dx = (x - arr[offset]) / (arr[offset + 1] - arr[offset]);

	// This assumes array values to be in the range [0,1) which is fine for our purposes because we use them as texture coordinates
	if (xFound)
		*xFound = (offset + 1 + dx) / arrCount;

	return offset;
}

class ImportanceSamplingEnvMaps : public ApplicationBase
{
	static constexpr uint32_t WIN_W = 2048;
	static constexpr uint32_t WIN_H = 1024;
	static constexpr uint32_t SC_IMG_COUNT = 3u;
	static constexpr uint32_t FRAMES_IN_FLIGHT = 5u;
	static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
	static constexpr size_t NBL_FRAMES_TO_AVERAGE = 100ull;

	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

	struct ShaderParameters
	{
		const uint32_t MaxDepthLog2 = 4; //5
		const uint32_t MaxSamplesLog2 = 10; //18
	} kShaderParameters;

public:
	nbl::core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
	nbl::core::smart_refctd_ptr<nbl::ui::IWindow> window;
	nbl::core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
	nbl::core::smart_refctd_ptr<nbl::video::IAPIConnection> apiConnection;
	nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface;
	nbl::core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
	nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
	nbl::video::IPhysicalDevice* physicalDevice;
	std::array<video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> queues;
	nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
	nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, CommonAPI::InitOutput::MaxSwapChainImageCount> fbo;
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxQueuesCount> commandPools;
	nbl::core::smart_refctd_ptr<nbl::system::ISystem> system;
	nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
	nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
	nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger;
	nbl::core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;

	nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
	core::smart_refctd_ptr<video::IDescriptorPool> descriptorPool;
	video::CDumbPresentationOracle oracle;

	core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FRAMES_IN_FLIGHT];

	CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
	CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;
	Camera camera = Camera(vectorSIMDf(0, 0, 0), vectorSIMDf(0, 0, 0), matrix4SIMD());

	core::smart_refctd_ptr<IGPUGraphicsPipeline> gpuEnvmapPipeline;
	core::smart_refctd_ptr<IGPUMeshBuffer> gpuEnvmapMeshBuffer;
	video::IGPUFramebuffer* HDRFramebuffer;
	core::smart_refctd_ptr<video::IGPUBuffer> gpuubo;
	core::smart_refctd_ptr<IGPUDescriptorSet> uboDescriptorSet1;
	core::smart_refctd_ptr<IGPUDescriptorSet> descriptorSet5;
	float envmapNormalizationFactor;

	bool ss = true;
	uint32_t acquiredNextFBO = {};
	int resourceIx = -1;

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

	nbl::video::IGPUFramebuffer* createHDRFramebuffer(asset::E_FORMAT colorFormat)
	{
		smart_refctd_ptr<IGPUImageView> gpuImageViewColorBuffer;
		{
			IGPUImage::SCreationParams imgInfo;
			imgInfo.format = colorFormat;
			imgInfo.type = IGPUImage::ET_2D;
			imgInfo.extent.width = WIN_W;
			imgInfo.extent.height = WIN_H;
			imgInfo.extent.depth = 1u;
			imgInfo.mipLevels = 1u;
			imgInfo.arrayLayers = 1u;
			imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
			imgInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);

			auto image = logicalDevice->createGPUImageOnDedMem(std::move(imgInfo), logicalDevice->getDeviceLocalGPUMemoryReqs());

			IGPUImageView::SCreationParams imgViewInfo;
			imgViewInfo.image = std::move(image);
			imgViewInfo.format = colorFormat;
			imgViewInfo.viewType = IGPUImageView::ET_2D;
			imgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
			imgViewInfo.subresourceRange.baseArrayLayer = 0u;
			imgViewInfo.subresourceRange.baseMipLevel = 0u;
			imgViewInfo.subresourceRange.layerCount = 1u;
			imgViewInfo.subresourceRange.levelCount = 1u;

			gpuImageViewColorBuffer = logicalDevice->createImageView(std::move(imgViewInfo));
		}

		// TODO:
		auto frameBuffer = logicalDevice->addFrameBuffer();
		frameBuffer->attach(video::EFAP_COLOR_ATTACHMENT0, std::move(gpuImageViewColorBuffer));

		return frameBuffer;
	}

	core::smart_refctd_ptr<IGPUImageView> getLUTGPUImageViewFromBuffer(core::smart_refctd_ptr<ICPUBuffer> buffer, IImage::E_TYPE imageType, asset::E_FORMAT format, const asset::VkExtent3D& extent,
		IGPUImageView::E_TYPE imageViewType)
	{
		auto gpuBuffer = utilities->createFilledDeviceLocalGPUBufferOnDedMem(queues[CommonAPI::InitOutput::EQT_TRANSFER_UP], buffer->getSize(), buffer->getPointer());

		IGPUImage::SCreationParams params;
		params.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
		params.type = imageType;
		params.format = format;
		params.extent = extent;
		params.mipLevels = 1u;
		params.arrayLayers = 1u;
		params.samples = asset::ICPUImage::ESCF_1_BIT;

		IGPUImage::SBufferCopy region = {}; // defaults 
		region.imageSubresource = {}; // defaults
		region.imageSubresource.layerCount = 1u;
		region.imageExtent = params.extent;

		auto gpuImage = utilities->createFilledDeviceLocalGPUImageOnDedMem(queues[CommonAPI::InitOutput::EQT_TRANSFER_UP], std::move(params), gpuBuffer.get(), 1u, &region);

		IGPUImageView::SCreationParams viewParams;
		viewParams.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		viewParams.image = gpuImage;
		viewParams.viewType = imageViewType;
		viewParams.format = viewParams.image->getCreationParameters().format;
		viewParams.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0u);
		viewParams.subresourceRange.baseMipLevel = 0;
		viewParams.subresourceRange.levelCount = 1;
		viewParams.subresourceRange.baseArrayLayer = 0;
		viewParams.subresourceRange.layerCount = 1;

		return logicalDevice->createImageView(std::move(viewParams));
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

	APP_CONSTRUCTOR(ImportanceSamplingEnvMaps)

	void onAppInitialized_impl() override
	{
		CommonAPI::InitOutput initOutput;
		initOutput.window = core::smart_refctd_ptr(window);
		initOutput.system = core::smart_refctd_ptr(system);

		const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT);
		const video::ISurface::SFormat surfaceFormat(asset::EF_R8G8B8A8_SRGB, asset::ECP_SRGB, asset::EOTF_sRGB);

		CommonAPI::InitWithDefaultExt(initOutput, video::EAT_OPENGL_ES, "ImportanceSamplingEnvMaps", WIN_W, WIN_H, SC_IMG_COUNT, swapchainImageUsage, surfaceFormat, nbl::asset::EF_D32_SFLOAT);
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

		core::vectorSIMDf cameraPosition(-0.0889001, 0.678913, -4.01774);
		core::vectorSIMDf cameraTarget(1.80119, 0.515374, -0.410544);
		matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60.0f), float(WIN_W) / WIN_H, 0.03125f, 200.0f);
		camera = Camera(cameraPosition, cameraTarget, projectionMatrix, 10.f, 1.f);

		descriptorPool = createDescriptorPool(1u);

		video::IGPUBuffer::SCreationParams gpuUBOParams;
		gpuUBOParams.canUpdateSubRange = true;
		gpuUBOParams.usage = asset::IBuffer::EUF_UNIFORM_BUFFER_BIT;
		gpuUBOParams.sharingMode = asset::E_SHARING_MODE::ESM_CONCURRENT;
		gpuUBOParams.queueFamilyIndexCount = 0u;
		gpuUBOParams.queueFamilyIndices = nullptr;
		gpuubo = logicalDevice->createDeviceLocalGPUBufferOnDedMem(gpuUBOParams, sizeof(SBasicViewParameters));

		const char* envmapPath = "../../media/envmap/envmap_1.exr";
		core::smart_refctd_ptr<IGPUImageView> envmapImageView = nullptr;
		core::smart_refctd_ptr<IGPUImageView> phiPdfLUTImageView = nullptr;
		core::smart_refctd_ptr<IGPUImageView> thetaLUTImageView = nullptr;

		{
			IAssetLoader::SAssetLoadParams lp(0ull, nullptr, IAssetLoader::ECF_DONT_CACHE_REFERENCES);
			auto envmapImageBundle = assetManager->getAsset(envmapPath, lp);
			auto envmapImage = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(*envmapImageBundle.getContents().begin());
			const uint32_t channelCount = getFormatChannelCount(envmapImage->getCreationParameters().format);

			auto luminancePdfBuffer = computeLuminancePdf(envmapImage, &envmapNormalizationFactor);

			ICPUImageView::SCreationParams viewParams;
			viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
			viewParams.image = envmapImage;
			viewParams.format = viewParams.image->getCreationParameters().format;
			viewParams.viewType = IImageView<ICPUImage>::ET_2D;
			viewParams.subresourceRange.baseArrayLayer = 0u;
			viewParams.subresourceRange.layerCount = 1u;
			viewParams.subresourceRange.baseMipLevel = 0u;
			viewParams.subresourceRange.levelCount = 1u;

			auto cpuEnvmapImageView = ICPUImageView::create(std::move(viewParams));
			cpu2gpuParams.beginCommandBuffers();
			envmapImageView = cpu2gpu.getGPUObjectsFromAssets(&cpuEnvmapImageView.get(), &cpuEnvmapImageView.get() + 1u, cpu2gpuParams)->front();
			cpu2gpuParams.waitForCreationToComplete();

			const core::vector2d<uint32_t> pdfDomainExtent = { envmapImage->getCreationParameters().extent.width, envmapImage->getCreationParameters().extent.height };

			core::smart_refctd_ptr<ICPUImage> conditionalCdfImage = nullptr;
			core::smart_refctd_ptr<ICPUBuffer> conditionalIntegrals = nullptr;
			{
				// Create ICPUImage from the buffer for the input image to the SAT filter
				auto luminanceImageParams = envmapImage->getCreationParameters();
				luminanceImageParams.format = EF_R64_SFLOAT;
				luminanceImageParams.extent = { pdfDomainExtent.X, pdfDomainExtent.Y, 1 };

				auto luminanceImageRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1ull);
				luminanceImageRegions->begin()->bufferOffset = 0ull;
				luminanceImageRegions->begin()->bufferRowLength = luminanceImageParams.extent.width;
				luminanceImageRegions->begin()->bufferImageHeight = 0u;
				luminanceImageRegions->begin()->imageSubresource = {};
				luminanceImageRegions->begin()->imageSubresource.layerCount = 1u;
				luminanceImageRegions->begin()->imageOffset = { 0, 0, 0 };
				luminanceImageRegions->begin()->imageExtent = { luminanceImageParams.extent.width, luminanceImageParams.extent.height, 1 };

				core::smart_refctd_ptr<ICPUImage> luminanceImage = ICPUImage::create(std::move(luminanceImageParams));
				luminanceImage->setBufferAndRegions(core::smart_refctd_ptr(luminancePdfBuffer), luminanceImageRegions);

				// Create out image
				const size_t conditionalCdfBufferSize = pdfDomainExtent.X * pdfDomainExtent.Y * sizeof(double);
				core::smart_refctd_ptr<ICPUBuffer> conditionalCdfBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(conditionalCdfBufferSize);
				memset(conditionalCdfBuffer->getPointer(), 0, conditionalCdfBufferSize);

				auto conditionalCdfImageParams = luminanceImage->getCreationParameters();
				conditionalCdfImageParams.format = EF_R64_SFLOAT;

				auto conditionalCdfImageRegions(luminanceImageRegions);
				conditionalCdfImage = ICPUImage::create(std::move(conditionalCdfImageParams));
				conditionalCdfImage->setBufferAndRegions(std::move(conditionalCdfBuffer), conditionalCdfImageRegions);

				// Set up the filter state
				SATFilter sum_filter;
				SATFilter::state_type state;

				state.inImage = luminanceImage.get();
				state.outImage = conditionalCdfImage.get();
				state.inOffset = { 0, 0, 0 };
				state.inBaseLayer = 0;
				state.outOffset = { 0, 0, 0 };
				state.outBaseLayer = 0;
				state.extent = luminanceImage->getCreationParameters().extent;
				state.layerCount = luminanceImage->getCreationParameters().arrayLayers;
				state.scratchMemoryByteSize = state.getRequiredScratchByteSize(state.inImage, state.extent);
				state.scratchMemory = reinterpret_cast<uint8_t*>(_NBL_ALIGNED_MALLOC(state.scratchMemoryByteSize, 32));
				state.axesToSum = ((0) << 2) | ((0) << 1) | ((1) << 0); // ZYX
				state.inMipLevel = 0;
				state.outMipLevel = 0;

				if (!sum_filter.execute(core::execution::par_unseq, &state))
					std::cout << "SAT filter failed for some reason" << std::endl;

				_NBL_ALIGNED_FREE(state.scratchMemory);

				// From the outImage you gotta extract integrals and normalize
				double* conditionalCdfPixel = (double*)conditionalCdfImage->getBuffer()->getPointer();

				conditionalIntegrals = core::make_smart_refctd_ptr<ICPUBuffer>(pdfDomainExtent.Y * sizeof(double));
				double* conditionalIntegralsPixel = (double*)conditionalIntegrals->getPointer();
				for (uint32_t y = 0; y < pdfDomainExtent.Y; ++y)
				{
					// printf("\n Conditional Integral[%d] = %f", y, conditionalCdfPixel[y * pdfDomainExtent.X + (pdfDomainExtent.X - 1)]);
					*conditionalIntegralsPixel++ = conditionalCdfPixel[y * pdfDomainExtent.X + (pdfDomainExtent.X - 1)];
				}

				conditionalCdfPixel = (double*)conditionalCdfImage->getBuffer()->getPointer();
				conditionalIntegralsPixel = (double*)conditionalIntegrals->getPointer();

				// now normalize
				for (uint32_t y = 0; y < pdfDomainExtent.Y; ++y)
				{
					for (uint32_t x = 0; x < pdfDomainExtent.X; ++x)
					{
						conditionalCdfPixel[y * pdfDomainExtent.X + x] /= conditionalIntegralsPixel[y];
					}
				}
			}

			core::smart_refctd_ptr<ICPUImage> marginalCdfImage = nullptr;
			double marginalIntegral = 0.0;
			{
				// Input: conditionalIntegrals
				// Create ICPUImage from the buffer for the input image to the SAT filter
				IImage::SCreationParams inParams;
				inParams.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
				inParams.type = IImage::ET_1D;
				inParams.format = asset::EF_R64_SFLOAT;
				inParams.extent = { pdfDomainExtent.Y, 1, 1 };
				inParams.mipLevels = 1u;
				inParams.arrayLayers = 1u;
				inParams.samples = asset::ICPUImage::ESCF_1_BIT;

				auto inImageRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1ull);
				inImageRegions->begin()->bufferOffset = 0ull;
				inImageRegions->begin()->bufferRowLength = inParams.extent.width;
				inImageRegions->begin()->bufferImageHeight = 0u;
				inImageRegions->begin()->imageSubresource = {};
				inImageRegions->begin()->imageSubresource.layerCount = 1u;
				inImageRegions->begin()->imageOffset = { 0, 0, 0 };
				inImageRegions->begin()->imageExtent = { inParams.extent.width, inParams.extent.height, inParams.extent.depth };

				core::smart_refctd_ptr<ICPUImage> inImage = ICPUImage::create(std::move(inParams));
				inImage->setBufferAndRegions(core::smart_refctd_ptr(conditionalIntegrals), inImageRegions);

				// Ouput: 1d cdf of conditionalIntegrals
				// Create out image
				core::smart_refctd_ptr<ICPUBuffer> marginalCdfBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(conditionalIntegrals->getSize());
				memset(marginalCdfBuffer->getPointer(), 0, marginalCdfBuffer->getSize());

				auto marginalCdfImageParams = inImage->getCreationParameters();
				marginalCdfImageParams.format = EF_R64_SFLOAT;

				auto marginalCdfImageRegions(inImageRegions);
				marginalCdfImage = ICPUImage::create(std::move(marginalCdfImageParams));
				marginalCdfImage->setBufferAndRegions(std::move(marginalCdfBuffer), marginalCdfImageRegions);

				// Set up the filter state
				SATFilter sum_filter;
				SATFilter::state_type state;

				state.inImage = inImage.get();
				state.outImage = marginalCdfImage.get();
				state.inOffset = { 0, 0, 0 };
				state.inBaseLayer = 0;
				state.outOffset = { 0, 0, 0 };
				state.outBaseLayer = 0;
				state.extent = inImage->getCreationParameters().extent;
				state.layerCount = inImage->getCreationParameters().arrayLayers;
				state.scratchMemoryByteSize = state.getRequiredScratchByteSize(state.inImage, state.extent);
				state.scratchMemory = reinterpret_cast<uint8_t*>(_NBL_ALIGNED_MALLOC(state.scratchMemoryByteSize, 32));
				state.axesToSum = ((0) << 2) | ((0) << 1) | ((1) << 0); // ZYX
				state.inMipLevel = 0;
				state.outMipLevel = 0;

				if (!sum_filter.execute(core::execution::par_unseq, &state))
					std::cout << "SAT filter failed for some reason" << std::endl;

				_NBL_ALIGNED_FREE(state.scratchMemory);

				// From the outImage you gotta extract integral and normalize
				double* marginalCdfPixel = (double*)marginalCdfImage->getBuffer()->getPointer();

				marginalIntegral = marginalCdfPixel[pdfDomainExtent.Y - 1];

				// now normalize
				for (uint32_t y = 0; y < pdfDomainExtent.Y; ++y)
				{
					// printf("\n MarginalCDFPixel[%d] = %f", y, marginalCdfPixel[y]);
					marginalCdfPixel[y] /= marginalIntegral;
				}
			}

			for (uint32_t i = 1; i < (marginalCdfImage->getBuffer()->getSize() / sizeof(double)); ++i)
				assert(((double*)marginalCdfImage->getBuffer()->getPointer())[i] > ((double*)marginalCdfImage->getBuffer()->getPointer())[i - 1]);

			// Computing LUTs

			const uint32_t phiPdfLUTChannelCount = 2u; // phi and pdf
			const size_t phiPdfLUTBufferSize = pdfDomainExtent.X * pdfDomainExtent.Y * phiPdfLUTChannelCount * sizeof(float);
			core::smart_refctd_ptr<ICPUBuffer> phiPdfLUTBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(phiPdfLUTBufferSize);
			memset(phiPdfLUTBuffer->getPointer(), 0, phiPdfLUTBufferSize);

			const uint32_t thetaLUTChannelCount = 1u; // theta
			const size_t thetaLUTBufferSize = pdfDomainExtent.Y * thetaLUTChannelCount * sizeof(float);
			core::smart_refctd_ptr<ICPUBuffer> thetaLUTBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(thetaLUTBufferSize);
			memset(thetaLUTBuffer->getPointer(), 0, thetaLUTBufferSize);

			float* phiPdfLUTPixel = (float*)phiPdfLUTBuffer->getPointer();
			float* thetaLUTPixel = (float*)thetaLUTBuffer->getPointer();

			core::vector2d<double> xi(0.0, 0.0);
			core::vector2d<double> xiRemapped = { 0.0, 0.0 };
			for (uint32_t y = 0; y < pdfDomainExtent.Y; ++y)
			{
				xi.Y = (y + 0.5) / (double)pdfDomainExtent.Y;

				int32_t yoffset = bisectionSearch((double*)marginalCdfImage->getBuffer()->getPointer(), pdfDomainExtent.Y, xi.Y, &xiRemapped.Y);
				const uint32_t rowToSample = (uint32_t)(yoffset + 1);
				assert(rowToSample < pdfDomainExtent.Y);
				double marginalPdf = ((double*)conditionalIntegrals->getPointer())[rowToSample] / marginalIntegral;

				const double theta = xiRemapped.Y * core::PI<double>();
				*thetaLUTPixel++ = (float)theta;

				for (uint32_t x = 0; x < pdfDomainExtent.X; ++x)
				{
					xi.X = (x + 0.5) / (double)pdfDomainExtent.X;

					const int32_t xoffset = bisectionSearch((double*)conditionalCdfImage->getBuffer()->getPointer() + rowToSample * pdfDomainExtent.X, pdfDomainExtent.X, xi.X, &xiRemapped.X);
					const uint32_t colToSample = (uint32_t)(xoffset + 1);
					assert(colToSample < pdfDomainExtent.X);
					const double conditionalPdf = ((double*)luminancePdfBuffer->getPointer())[rowToSample * pdfDomainExtent.X + colToSample] / ((double*)conditionalIntegrals->getPointer())[rowToSample];

					const double phi = xiRemapped.X * 2.0 * core::PI<double>();
					const double pdf = (core::sin(theta) == 0.0) ? 0.0 : (marginalPdf * conditionalPdf) / (2.0 * core::PI<double>() * core::PI<double>() * core::sin(theta));

					*phiPdfLUTPixel++ = (float)phi;
					*phiPdfLUTPixel++ = (float)pdf;
				}
			}

			phiPdfLUTImageView = getLUTGPUImageViewFromBuffer(phiPdfLUTBuffer, IGPUImage::ET_2D, asset::EF_R32G32_SFLOAT, { pdfDomainExtent.X, pdfDomainExtent.Y, 1 }, IGPUImageView::ET_2D);
			thetaLUTImageView = getLUTGPUImageViewFromBuffer(thetaLUTBuffer, IGPUImage::ET_1D, asset::EF_R32_SFLOAT, { pdfDomainExtent.Y, 1, 1 }, IGPUImageView::ET_1D);
		}

		smart_refctd_ptr<IGPUBufferView> gpuSequenceBufferView;
		{
			const uint32_t MaxDimensions = 3u << kShaderParameters.MaxDepthLog2;
			const uint32_t MaxSamples = 1u << kShaderParameters.MaxSamplesLog2;

			auto sampleSequence = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(uint32_t) * MaxDimensions * MaxSamples);

			core::OwenSampler sampler(MaxDimensions, 0xdeadbeefu);
			//core::SobolSampler sampler(MaxDimensions);

			auto out = reinterpret_cast<uint32_t*>(sampleSequence->getPointer());
			for (auto dim = 0u; dim < MaxDimensions; dim++)
			{
				for (uint32_t i = 0; i < MaxSamples; i++)
				{
					out[i * MaxDimensions + dim] = sampler.sample(dim, i);
				}
			}

			auto gpuSequenceBuffer = utilities->createFilledDeviceLocalGPUBufferOnDedMem(queues[CommonAPI::InitOutput::EQT_TRANSFER_UP], sampleSequence->getSize(), sampleSequence->getPointer());
			auto gpuSequenceBuffer = cpu2gpu.getGPUObjectsFromAssets(&sampleSequence, &sampleSequence + 1u, cpu2gpuParams)->front()->getBuffer();
			gpuSequenceBufferView = logicalDevice->createBufferView(gpuSequenceBuffer.get(), asset::EF_R32G32B32_UINT);
		}

		smart_refctd_ptr<IGPUImageView> gpuScrambleImageView;
		{
			IGPUImage::SCreationParams imgParams;
			imgParams.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);
			imgParams.type = IImage::ET_2D;
			imgParams.format = EF_R32G32_UINT;
			imgParams.extent = { WIN_W,WIN_H,1u };
			imgParams.mipLevels = 1u;
			imgParams.arrayLayers = 1u;
			imgParams.samples = IImage::ESCF_1_BIT;

			IGPUImage::SBufferCopy region;
			region.imageExtent = imgParams.extent;
			region.imageSubresource.layerCount = 1u;

			constexpr auto ScrambleStateChannels = 2u;
			const auto renderPixelCount = imgParams.extent.width * imgParams.extent.height;
			core::vector<uint32_t> random(renderPixelCount * ScrambleStateChannels);
			{
				core::RandomSampler rng(0xbadc0ffeu);
				for (auto& pixel : random)
					pixel = rng.nextSample();
			}
			cpu2gpuParams.beginCommandBuffers();
			auto buffer = utilities->createFilledDeviceLocalGPUBufferOnDedMem(queues[CommonAPI::InitOutput::EQT_TRANSFER_UP], random.size() * sizeof(uint32_t), random.data());
			cpu2gpuParams.waitForCreationToComplete();

			IGPUImageView::SCreationParams viewParams;
			viewParams.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
			viewParams.image = utilities->createFilledDeviceLocalGPUImageOnDedMem(queues[CommonAPI::InitOutput::EQT_TRANSFER_UP], std::move(imgParams), buffer.get(), 1u, &region);
			viewParams.viewType = IGPUImageView::ET_2D;
			viewParams.format = EF_R32G32_UINT;
			viewParams.subresourceRange.levelCount = 1u;
			viewParams.subresourceRange.layerCount = 1u;
			gpuScrambleImageView = logicalDevice->createImageView(std::move(viewParams));
		}

		auto fullScreenTriangle = ext::FullScreenTriangle::createProtoPipeline(cpu2gpuParams);

		core::smart_refctd_ptr<IGPUDescriptorSetLayout> gpuDescriptorSetLayout1 = nullptr;
		core::smart_refctd_ptr<IGPUDescriptorSetLayout> gpuDescriptorSetLayout5 = nullptr;
		{
			IGPUDescriptorSetLayout::SBinding uboBinding{ 0, asset::EDT_UNIFORM_BUFFER, 1u, IGPUShader::ESS_FRAGMENT, nullptr };
			gpuDescriptorSetLayout1 = logicalDevice->createDescriptorSetLayout(&uboBinding, &uboBinding + 1u);

			constexpr uint32_t descriptorCount = 5u;
			IGPUDescriptorSetLayout::SBinding descriptorSet5Bindings[descriptorCount] =
			{
				{ 0u, EDT_COMBINED_IMAGE_SAMPLER, 1u, IGPUShader::ESS_FRAGMENT, nullptr },
				{ 1u, EDT_UNIFORM_TEXEL_BUFFER, 1u, IGPUShader::ESS_FRAGMENT, nullptr },
				{ 2u, EDT_COMBINED_IMAGE_SAMPLER, 1u, IGPUShader::ESS_FRAGMENT, nullptr },
				{ 3u, EDT_COMBINED_IMAGE_SAMPLER, 1u, IGPUShader::ESS_FRAGMENT, nullptr },
				{ 4u, EDT_COMBINED_IMAGE_SAMPLER, 1u, IGPUShader::ESS_FRAGMENT, nullptr }
			};
			gpuDescriptorSetLayout5 = logicalDevice->createDescriptorSetLayout(descriptorSet5Bindings, descriptorSet5Bindings + descriptorCount);
		}

		const SPushConstantRange pcRange =
		{
			IShader::ESS_FRAGMENT,
			0u,
			sizeof(float)
		};

		auto createGpuResources = [&](std::string pathToShader) -> std::pair<core::smart_refctd_ptr<video::IGPUGraphicsPipeline>, core::smart_refctd_ptr<video::IGPUMeshBuffer>>
		{
			auto cpuFragmentSpecializedShader = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(assetManager->getAsset(pathToShader, {}).getContents().begin()[0]);
			ISpecializedShader::SInfo info = cpuFragmentSpecializedShader->getSpecializationInfo();
			info.m_backingBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(sizeof(ShaderParameters));
			memcpy(info.m_backingBuffer->getPointer(), &kShaderParameters, sizeof(ShaderParameters));
			info.m_entries = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ISpecializedShader::SInfo::SMapEntry>>(2u);
			for (uint32_t i = 0; i < 2; i++)
				info.m_entries->operator[](i) = { i,i * sizeof(uint32_t),sizeof(uint32_t) };
			cpuFragmentSpecializedShader->setSpecializationInfo(std::move(info));

			cpu2gpuParams.beginCommandBuffers();
			auto gpuFragmentSpecialedShader = cpu2gpu.getGPUObjectsFromAssets(&cpuFragmentSpecializedShader.get(), &cpuFragmentSpecializedShader.get() + 1, cpu2gpuParams)->front();
			cpu2gpuParams.waitForCreationToComplete();
			IGPUSpecializedShader* shaders[2] = { std::get<0>(fullScreenTriangle).get(), gpuFragmentSpecialedShader.get() };

			// auto gpuPipelineLayout = driver->createPipelineLayout(nullptr, nullptr, nullptr, core::smart_refctd_ptr(gpuDescriptorSetLayout1), nullptr, core::smart_refctd_ptr(gpuDescriptorSetLayout5));
			auto gpuPipelineLayout = logicalDevice->createPipelineLayout(&pcRange, &pcRange + 1, nullptr, core::smart_refctd_ptr(gpuDescriptorSetLayout1), nullptr, core::smart_refctd_ptr(gpuDescriptorSetLayout5));

			asset::SBlendParams blendParams;
			SRasterizationParams rasterParams;
			rasterParams.faceCullingMode = EFCM_NONE;
			rasterParams.depthCompareOp = ECO_ALWAYS;
			rasterParams.minSampleShading = 1.f;
			rasterParams.depthWriteEnable = false;
			rasterParams.depthTestEnable = false;

			auto gpuPipeline = logicalDevice->createRenderpassIndependentPipeline(
				nullptr, std::move(gpuPipelineLayout),
				shaders, shaders + sizeof(shaders) / sizeof(IGPUSpecializedShader*),
				std::get<SVertexInputParams>(fullScreenTriangle), blendParams, std::get<SPrimitiveAssemblyParams>(fullScreenTriangle), rasterParams);

			nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
			graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(const_cast<video::IGPURenderpassIndependentPipeline*>(gpuPipeline.get()));
			graphicsPipelineParams.renderpass = core::smart_refctd_ptr(renderpass);

			auto gpuGraphicsPipeline = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));

			SBufferBinding<IGPUBuffer> idxBinding{ 0ull, nullptr };
			core::smart_refctd_ptr<video::IGPUMeshBuffer> gpuMeshBuffer = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(core::smart_refctd_ptr(gpuPipeline), nullptr, nullptr, std::move(idxBinding));
			{
				gpuMeshBuffer->setIndexCount(3u);
			}

			return { gpuGraphicsPipeline, gpuMeshBuffer };
		};

		const char* fragment_shader_path = "../fullscreen.frag";
		auto gpuEnvmapResources = createGpuResources(fragment_shader_path);
		gpuEnvmapPipeline = gpuEnvmapResources.first;
		gpuEnvmapMeshBuffer = gpuEnvmapResources.second;

		// Create and update DS
		uboDescriptorSet1 = logicalDevice->createDescriptorSet(descriptorPool.get(), core::smart_refctd_ptr(gpuDescriptorSetLayout1));
		{
			video::IGPUDescriptorSet::SWriteDescriptorSet uboWriteDescriptorSet;
			uboWriteDescriptorSet.dstSet = uboDescriptorSet1.get();
			uboWriteDescriptorSet.binding = 0;
			uboWriteDescriptorSet.count = 1u;
			uboWriteDescriptorSet.arrayElement = 0u;
			uboWriteDescriptorSet.descriptorType = asset::EDT_UNIFORM_BUFFER;
			video::IGPUDescriptorSet::SDescriptorInfo info;
			{
				info.desc = gpuubo;
				info.buffer.offset = 0ull;
				info.buffer.size = sizeof(SBasicViewParameters);
			}
			uboWriteDescriptorSet.info = &info;

			logicalDevice->updateDescriptorSets(1u, &uboWriteDescriptorSet, 0u, nullptr);
		}

		descriptorSet5 = logicalDevice->createDescriptorSet(descriptorPool.get(), core::smart_refctd_ptr(gpuDescriptorSetLayout5));
		{
			constexpr auto kDescriptorCount = 5;

			IGPUDescriptorSet::SDescriptorInfo descriptorInfos[kDescriptorCount];
			descriptorInfos[0].desc = envmapImageView;
			{
				ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
				descriptorInfos[0].image.sampler = logicalDevice->createSampler(samplerParams);
				descriptorInfos[0].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
			}

			descriptorInfos[1].desc = gpuSequenceBufferView;

			descriptorInfos[2].desc = gpuScrambleImageView;
			{
				ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_INT_OPAQUE_BLACK, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, ISampler::ESMM_NEAREST, 0u, false, ECO_ALWAYS };
				descriptorInfos[2].image.sampler = logicalDevice->createSampler(samplerParams);
				descriptorInfos[2].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
			}

			descriptorInfos[3].desc = phiPdfLUTImageView;
			{
				ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, ISampler::ESMM_NEAREST, 0u, false, ECO_ALWAYS };
				descriptorInfos[3].image.sampler = logicalDevice->createSampler(samplerParams);
				descriptorInfos[3].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
			}

			descriptorInfos[4].desc = thetaLUTImageView;
			{
				ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, ISampler::ESMM_NEAREST, 0u, false, ECO_ALWAYS };
				descriptorInfos[4].image.sampler = logicalDevice->createSampler(samplerParams);
				descriptorInfos[4].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
			}

			IGPUDescriptorSet::SWriteDescriptorSet descriptorSetWrites[kDescriptorCount];
			for (auto i = 0; i < kDescriptorCount; i++)
			{
				descriptorSetWrites[i].dstSet = descriptorSet5.get();
				descriptorSetWrites[i].binding = i;
				descriptorSetWrites[i].arrayElement = 0u;
				descriptorSetWrites[i].count = 1u;
				descriptorSetWrites[i].info = descriptorInfos + i;
			}
			descriptorSetWrites[0].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
			descriptorSetWrites[1].descriptorType = EDT_UNIFORM_TEXEL_BUFFER;
			descriptorSetWrites[2].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
			descriptorSetWrites[3].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
			descriptorSetWrites[4].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;

			logicalDevice->updateDescriptorSets(kDescriptorCount, descriptorSetWrites, 0u, nullptr);
		}

		HDRFramebuffer = createHDRFramebuffer(asset::EF_R32G32B32A32_SFLOAT);

		oracle.reportBeginFrameRecord();
		logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, commandBuffers);

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
		{
			imageAcquire[i] = logicalDevice->createSemaphore();
			renderFinished[i] = logicalDevice->createSemaphore();
		}
	}

	void onAppTerminated_impl() override
	{
		const core::vectorSIMDf& last_cam_pos = camera.getPosition();
		const core::vectorSIMDf& last_cam_target = camera.getTarget();
		std::cout << "Last camera position: (" << last_cam_pos.X << ", " << last_cam_pos.Y << ", " << last_cam_pos.Z << ")" << std::endl;
		std::cout << "Last camera target: (" << last_cam_target.X << ", " << last_cam_target.Y << ", " << last_cam_target.Z << ")" << std::endl;
	}

	void workLoopBody() override
	{
		++resourceIx;
		if (resourceIx >= FRAMES_IN_FLIGHT)
			resourceIx = 0;

		auto& commandBuffer = commandBuffers[resourceIx];
		auto& fence = frameComplete[resourceIx];
		if (fence)
			logicalDevice->blockForFences(1u, &fence.get());
		else
			fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

		commandBuffer->reset(nbl::video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
		commandBuffer->begin(0);

		const auto nextPresentationTimestamp = oracle.acquireNextImage(swapchain.get(), imageAcquire[resourceIx].get(), nullptr, &acquiredNextFBO);
		{
			inputSystem->getDefaultMouse(&mouse);
			inputSystem->getDefaultKeyboard(&keyboard);

			camera.beginInputProcessing(nextPresentationTimestamp);
			mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); }, logger.get());
			keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); }, logger.get());
			camera.endInputProcessing(nextPresentationTimestamp);
		}

		const auto& viewMatrix = camera.getViewMatrix();
		const auto& viewProjectionMatrix = camera.getConcatenatedMatrix();

		asset::SViewport viewport;
		viewport.minDepth = 1.f;
		viewport.maxDepth = 0.f;
		viewport.x = 0u;
		viewport.y = 0u;
		viewport.width = WIN_W;
		viewport.height = WIN_H;
		commandBuffer->setViewport(0u, 1u, &viewport);

		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = { WIN_W, WIN_H };
		commandBuffer->setScissor(0u, 1u, &scissor);

		core::matrix3x4SIMD modelMatrix;
		modelMatrix.setTranslation(nbl::core::vectorSIMDf(0, 0, 0, 0));
		core::matrix4SIMD mvp = core::concatenateBFollowedByA(viewProjectionMatrix, modelMatrix);

		SBasicViewParameters uboContents;
		memcpy(uboContents.MVP, mvp.pointer(), sizeof(float) * 4u * 4u);
		memcpy(uboContents.MV, viewProjectionMatrix.pointer(), sizeof(float) * 4u * 3u);
		memcpy(uboContents.NormalMat, viewProjectionMatrix.pointer(), sizeof(float) * 4u * 3u);

		commandBuffer->updateBuffer(gpuubo.get(), 0ull, sizeof(uboContents), &uboContents);

		nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
		{
			VkRect2D area;
			area.offset = { 0,0 };
			area.extent = { WIN_W, WIN_H };
			asset::SClearValue clear[2] = {};
			clear[0].color.float32[0] = 1.f;
			clear[0].color.float32[1] = 1.f;
			clear[0].color.float32[2] = 1.f;
			clear[0].color.float32[3] = 1.f;
			clear[1].depthStencil.depth = 0.f;

			beginInfo.clearValueCount = 2u;
			beginInfo.framebuffer = fbo[acquiredNextFBO];
			beginInfo.renderpass = renderpass;
			beginInfo.renderArea = area;
			beginInfo.clearValues = clear;
		}

		commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);

		commandBuffer->bindGraphicsPipeline(gpuEnvmapPipeline.get()); 
		commandBuffer->bindDescriptorSets(EPBP_GRAPHICS, gpuEnvmapPipeline->getRenderpassIndependentPipeline()->getLayout(), 1u, 1u, &uboDescriptorSet1.get(), 0u);
		commandBuffer->bindDescriptorSets(EPBP_GRAPHICS, gpuEnvmapPipeline->getRenderpassIndependentPipeline()->getLayout(), 3u, 1u, &descriptorSet5.get(), 0u);
		commandBuffer->pushConstants(gpuEnvmapPipeline->getRenderpassIndependentPipeline()->getLayout(), video::IGPUShader::ESS_FRAGMENT, 0u, sizeof(float), &envmapNormalizationFactor);
		commandBuffer->drawMeshBuffer(gpuEnvmapMeshBuffer.get());

		// TODO:
		// driver->setRenderTarget(nullptr, false);
		// driver->blitRenderTargets(HDRFramebuffer, nullptr, false, false);

		commandBuffer->endRenderPass();
		commandBuffer->end();
		
		if (ss)
		{
			//TODO:
			//ext::ScreenShot::createScreenShot(device, HDRFramebuffer->getAttachment(video::EFAP_COLOR_ATTACHMENT0), "screenshot.exr", asset::EIL_PRESENT_SRC, static_cast<asset::E_ACCESS_FLAGS>(0u));
			//ss = false;
		}
	}

	bool keepRunning() override
	{
		return windowCb->isWindowOpen();
	}
};
NBL_COMMON_API_MAIN(ImportanceSamplingEnvMaps)