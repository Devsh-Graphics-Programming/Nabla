// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>
#include "../../../include/nbl/asset/filters/CSummedAreaTableImageFilter.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "glm/glm.hpp"
#include "../common/QToQuitEventReceiver.h"

using namespace nbl;
using namespace asset;
using namespace core;
using namespace video;

using SATFilter = CSummedAreaTableImageFilter<false>;

static core::smart_refctd_ptr<ICPUBuffer> computeLuminancePdf(smart_refctd_ptr<ICPUImage> envmap)
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

static core::smart_refctd_ptr<IGPUImageView> getLUTGPUImageViewFromBuffer(core::smart_refctd_ptr<ICPUBuffer> buffer, IImage::E_TYPE imageType, asset::E_FORMAT format, const asset::VkExtent3D& extent,
	IGPUImageView::E_TYPE imageViewType, video::IVideoDriver* driver)
{
	auto gpuBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(buffer->getSize(), buffer->getPointer());

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

	auto gpuImage = driver->createFilledDeviceLocalGPUImageOnDedMem(std::move(params), gpuBuffer.get(), 1u, &region);

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

	return driver->createGPUImageView(std::move(viewParams));
}

static nbl::video::IFrameBuffer* createHDRFramebuffer(IVideoDriver* driver, asset::E_FORMAT colorFormat)
{
	smart_refctd_ptr<IGPUImageView> gpuImageViewColorBuffer;
	{
		IGPUImage::SCreationParams imgInfo;
		imgInfo.format = colorFormat;
		imgInfo.type = IGPUImage::ET_2D;
		imgInfo.extent.width = driver->getScreenSize().Width;
		imgInfo.extent.height = driver->getScreenSize().Height;
		imgInfo.extent.depth = 1u;
		imgInfo.mipLevels = 1u;
		imgInfo.arrayLayers = 1u;
		imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
		imgInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);

		auto image = driver->createGPUImageOnDedMem(std::move(imgInfo), driver->getDeviceLocalGPUMemoryReqs());

		IGPUImageView::SCreationParams imgViewInfo;
		imgViewInfo.image = std::move(image);
		imgViewInfo.format = colorFormat;
		imgViewInfo.viewType = IGPUImageView::ET_2D;
		imgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		imgViewInfo.subresourceRange.baseArrayLayer = 0u;
		imgViewInfo.subresourceRange.baseMipLevel = 0u;
		imgViewInfo.subresourceRange.layerCount = 1u;
		imgViewInfo.subresourceRange.levelCount = 1u;

		gpuImageViewColorBuffer = driver->createGPUImageView(std::move(imgViewInfo));
	}

	auto frameBuffer = driver->addFrameBuffer();
	frameBuffer->attach(video::EFAP_COLOR_ATTACHMENT0, std::move(gpuImageViewColorBuffer));

	return frameBuffer;
}

struct ShaderParameters
{
	const uint32_t MaxDepthLog2 = 4; //5
	const uint32_t MaxSamplesLog2 = 10; //18
} kShaderParameters;

int main()
{
	nbl::SIrrlichtCreationParameters params;
	params.Bits = 24;
	params.ZBufferBits = 24;
	params.DriverType = video::EDT_OPENGL;
	params.WindowSize = core::dimension2d<uint32_t>(2048, 1024);
	params.Fullscreen = false;
	params.Vsync = false;
	params.Doublebuffer = true;
	params.Stencilbuffer = false;
	params.AuxGLContexts = 16;
	auto device = createDeviceEx(params);

	if (!device)
		return 1;

	video::IVideoDriver* driver = device->getVideoDriver();
	IAssetManager* assetManager = device->getAssetManager();
	auto filesystem = device->getFileSystem();
	auto glslc = assetManager->getGLSLCompiler();
	auto sceneManager = device->getSceneManager();
	auto geometryCreator = device->getAssetManager()->getGeometryCreator();

	device->getCursorControl()->setVisible(false);

	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);

	scene::ICameraSceneNode* camera = sceneManager->addCameraSceneNodeFPS(0, 100.0f, 0.001f);
	camera->setLeftHanded(false);

	camera->setPosition(core::vector3df(-0.0889001, 0.678913, -4.01774));
	camera->setTarget(core::vector3df(1.80119, 0.515374, -0.410544));
	camera->setNearValue(0.03125f);
	camera->setFarValue(200.0f);
	camera->setFOV(core::radians(60.f));

	sceneManager->setActiveCamera(camera);

	auto gpuubo = driver->createDeviceLocalGPUBufferOnDedMem(sizeof(SBasicViewParameters));

	const char* envmapPath = "../../media/envmap/envmap_1.exr";
	core::smart_refctd_ptr<IGPUImageView> envmapImageView = nullptr;
	core::smart_refctd_ptr<IGPUImageView> phiPdfLUTImageView = nullptr;
	core::smart_refctd_ptr<IGPUImageView> thetaLUTImageView = nullptr;
	{
		IAssetLoader::SAssetLoadParams lp(0ull, nullptr, IAssetLoader::ECF_DONT_CACHE_REFERENCES);
		auto envmapImageBundle = assetManager->getAsset(envmapPath, lp);
		auto envmapImage = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(*envmapImageBundle.getContents().begin());
		const uint32_t channelCount = getFormatChannelCount(envmapImage->getCreationParameters().format);

		auto luminancePdfBuffer = computeLuminancePdf(envmapImage);

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
		envmapImageView = driver->getGPUObjectsFromAssets(&cpuEnvmapImageView.get(), &cpuEnvmapImageView.get() + 1u)->front();

		const core::vector2d<uint32_t> pdfDomainExtent = { envmapImage->getCreationParameters().extent.width, envmapImage->getCreationParameters().extent.height };

		core::smart_refctd_ptr<ICPUImage> conditionalCdfImage = nullptr;
		core::smart_refctd_ptr<ICPUBuffer> conditionalIntegrals = nullptr;
		{
			// Create ICPUImage from the buffer for the input image to the SAT filter
			auto luminanceImageParams = envmapImage->getCreationParameters();
			luminanceImageParams.format = EF_R32_SFLOAT;
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

			if (!sum_filter.execute(std::execution::par_unseq, &state))
				std::cout << "SAT filter failed for some reason" << std::endl;

			_NBL_ALIGNED_FREE(state.scratchMemory);

			// From the outImage you gotta extract integrals and normalize
			double* conditionalCdfPixel = (double*)conditionalCdfImage->getBuffer()->getPointer();

			conditionalIntegrals = core::make_smart_refctd_ptr<ICPUBuffer>(pdfDomainExtent.Y * sizeof(double));
			double* conditionalIntegralsPixel = (double*)conditionalIntegrals->getPointer();
			for (uint32_t y = 0; y < pdfDomainExtent.Y; ++y)
				*conditionalIntegralsPixel++ = conditionalCdfPixel[y * pdfDomainExtent.X + (pdfDomainExtent.X - 1)];

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
			inParams.format = asset::EF_R32_SFLOAT;
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

			if (!sum_filter.execute(std::execution::par_unseq, &state))
				std::cout << "SAT filter failed for some reason" << std::endl;

			_NBL_ALIGNED_FREE(state.scratchMemory);

			// From the outImage you gotta extract integral and normalize
			double* marginalCdfPixel = (double*)marginalCdfImage->getBuffer()->getPointer();

			marginalIntegral = marginalCdfPixel[pdfDomainExtent.Y - 1];

			// now normalize
			for (uint32_t y = 0; y < pdfDomainExtent.Y; ++y)
				marginalCdfPixel[y] /= marginalIntegral;
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

			int32_t offset = bisectionSearch((double*)marginalCdfImage->getBuffer()->getPointer(), pdfDomainExtent.Y, xi.Y, &xiRemapped.Y);
			const uint32_t rowToSample = (uint32_t)offset + 1u;
			double marginalPdf = ((double*)conditionalIntegrals->getPointer())[offset + 1] / marginalIntegral;

			const double theta = xiRemapped.Y * core::PI<double>();
			*thetaLUTPixel++ = (float)theta;

			for (uint32_t x = 0; x < pdfDomainExtent.X; ++x)
			{
				xi.X = (x + 0.5) / (double)pdfDomainExtent.X;

				const int32_t offset = bisectionSearch((double*)conditionalCdfImage->getBuffer()->getPointer() + rowToSample * pdfDomainExtent.X, pdfDomainExtent.X, xi.X, &xiRemapped.X);
				const double conditionalPdf = ((double*)luminancePdfBuffer->getPointer())[rowToSample * pdfDomainExtent.X + offset + 1] / ((double*)conditionalIntegrals->getPointer())[rowToSample];

				const double phi = xiRemapped.X * 2.0 * core::PI<double>();
				const double pdf = (core::sin(theta) == 0.0) ? 0.0 : (marginalPdf * conditionalPdf) / (2.0 * core::PI<double>() * core::PI<double>() * core::sin(theta));



				*phiPdfLUTPixel++ = (float)phi;
				*phiPdfLUTPixel++ = (float)pdf;
			}
		}

		phiPdfLUTImageView = getLUTGPUImageViewFromBuffer(phiPdfLUTBuffer, IGPUImage::ET_2D, asset::EF_R32G32_SFLOAT, { pdfDomainExtent.X, pdfDomainExtent.Y, 1 }, IGPUImageView::ET_2D, driver);
		thetaLUTImageView = getLUTGPUImageViewFromBuffer(thetaLUTBuffer, IGPUImage::ET_1D, asset::EF_R32_SFLOAT, { pdfDomainExtent.Y, 1, 1 }, IGPUImageView::ET_1D, driver);

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
		auto gpuSequenceBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(sampleSequence->getSize(), sampleSequence->getPointer());
		gpuSequenceBufferView = driver->createGPUBufferView(gpuSequenceBuffer.get(), asset::EF_R32G32B32_UINT);
	}

	smart_refctd_ptr<IGPUImageView> gpuScrambleImageView;
	{
		IGPUImage::SCreationParams imgParams;
		imgParams.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);
		imgParams.type = IImage::ET_2D;
		imgParams.format = EF_R32G32_UINT;
		imgParams.extent = { params.WindowSize.Width,params.WindowSize.Height,1u };
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
		auto buffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(random.size() * sizeof(uint32_t), random.data());

		IGPUImageView::SCreationParams viewParams;
		viewParams.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		viewParams.image = driver->createFilledDeviceLocalGPUImageOnDedMem(std::move(imgParams), buffer.get(), 1u, &region);
		viewParams.viewType = IGPUImageView::ET_2D;
		viewParams.format = EF_R32G32_UINT;
		viewParams.subresourceRange.levelCount = 1u;
		viewParams.subresourceRange.layerCount = 1u;
		gpuScrambleImageView = driver->createGPUImageView(std::move(viewParams));
	}

	auto fullScreenTriangle = ext::FullScreenTriangle::createFullScreenTriangle(device->getAssetManager(), device->getVideoDriver());

	core::smart_refctd_ptr<IGPUDescriptorSetLayout> gpuDescriptorSetLayout1 = nullptr;
	core::smart_refctd_ptr<IGPUDescriptorSetLayout> gpuDescriptorSetLayout5 = nullptr;
	{
		IGPUDescriptorSetLayout::SBinding uboBinding{ 0, asset::EDT_UNIFORM_BUFFER, 1u, IGPUSpecializedShader::ESS_FRAGMENT, nullptr };
		gpuDescriptorSetLayout1 = driver->createGPUDescriptorSetLayout(&uboBinding, &uboBinding + 1u);

		constexpr uint32_t descriptorCount = 5u;
		IGPUDescriptorSetLayout::SBinding descriptorSet5Bindings[descriptorCount] =
		{
			{ 0u, EDT_COMBINED_IMAGE_SAMPLER, 1u, IGPUSpecializedShader::ESS_FRAGMENT, nullptr },
			{ 1u, EDT_UNIFORM_TEXEL_BUFFER, 1u, IGPUSpecializedShader::ESS_FRAGMENT, nullptr },
			{ 2u, EDT_COMBINED_IMAGE_SAMPLER, 1u, IGPUSpecializedShader::ESS_FRAGMENT, nullptr },
			{ 3u, EDT_COMBINED_IMAGE_SAMPLER, 1u, IGPUSpecializedShader::ESS_FRAGMENT, nullptr },
			{ 4u, EDT_COMBINED_IMAGE_SAMPLER, 1u, IGPUSpecializedShader::ESS_FRAGMENT, nullptr }
		};
		gpuDescriptorSetLayout5 = driver->createGPUDescriptorSetLayout(descriptorSet5Bindings, descriptorSet5Bindings + descriptorCount);
	}

	auto createGpuResources = [&](std::string pathToShader) -> std::pair<core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline>, core::smart_refctd_ptr<video::IGPUMeshBuffer>>
	{
		auto cpuFragmentSpecializedShader = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(assetManager->getAsset(pathToShader, {}).getContents().begin()[0]);
		ISpecializedShader::SInfo info = cpuFragmentSpecializedShader->getSpecializationInfo();
		info.m_backingBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(sizeof(ShaderParameters));
		memcpy(info.m_backingBuffer->getPointer(), &kShaderParameters, sizeof(ShaderParameters));
		info.m_entries = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ISpecializedShader::SInfo::SMapEntry>>(2u);
		for (uint32_t i = 0; i < 2; i++)
			info.m_entries->operator[](i) = { i,i * sizeof(uint32_t),sizeof(uint32_t) };
		cpuFragmentSpecializedShader->setSpecializationInfo(std::move(info));

		auto gpuFragmentSpecialedShader = driver->getGPUObjectsFromAssets(&cpuFragmentSpecializedShader.get(), &cpuFragmentSpecializedShader.get() + 1)->front();
		IGPUSpecializedShader* shaders[2] = { std::get<0>(fullScreenTriangle).get(), gpuFragmentSpecialedShader.get() };

		auto gpuPipelineLayout = driver->createGPUPipelineLayout(nullptr, nullptr, nullptr, core::smart_refctd_ptr(gpuDescriptorSetLayout1), nullptr, core::smart_refctd_ptr(gpuDescriptorSetLayout5));

		asset::SBlendParams blendParams;
		SRasterizationParams rasterParams;
		rasterParams.faceCullingMode = EFCM_NONE;
		rasterParams.depthCompareOp = ECO_ALWAYS;
		rasterParams.minSampleShading = 1.f;
		rasterParams.depthWriteEnable = false;
		rasterParams.depthTestEnable = false;

		auto gpuPipeline = driver->createGPURenderpassIndependentPipeline(
			nullptr, std::move(gpuPipelineLayout),
			shaders, shaders + sizeof(shaders) / sizeof(IGPUSpecializedShader*),
			std::get<SVertexInputParams>(fullScreenTriangle), blendParams, std::get<SPrimitiveAssemblyParams>(fullScreenTriangle), rasterParams);

		SBufferBinding<IGPUBuffer> idxBinding{ 0ull, nullptr };
		core::smart_refctd_ptr<video::IGPUMeshBuffer> gpuMeshBuffer = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(core::smart_refctd_ptr(gpuPipeline), nullptr, nullptr, std::move(idxBinding));
		{
			gpuMeshBuffer->setIndexCount(3u);
		}

		return { gpuPipeline, gpuMeshBuffer };
	};

	const char* fragment_shader_path = "../fullscreen.frag";
	auto gpuEnvmapResources = createGpuResources(fragment_shader_path);
	auto gpuEnvmapPipeline = gpuEnvmapResources.first;
	auto gpuEnvmapMeshBuffer = gpuEnvmapResources.second;

	// Create and update DS
	auto uboDescriptorSet1 = driver->createGPUDescriptorSet(core::smart_refctd_ptr(gpuDescriptorSetLayout1));
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

		driver->updateDescriptorSets(1u, &uboWriteDescriptorSet, 0u, nullptr);
	}

	auto descriptorSet5 = driver->createGPUDescriptorSet(core::smart_refctd_ptr(gpuDescriptorSetLayout5));
	{
		constexpr auto kDescriptorCount = 5;

		IGPUDescriptorSet::SDescriptorInfo descriptorInfos[kDescriptorCount];
		descriptorInfos[0].desc = envmapImageView;
		{
			ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
			descriptorInfos[0].image.sampler = driver->createGPUSampler(samplerParams);
			descriptorInfos[0].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
		}

		descriptorInfos[1].desc = gpuSequenceBufferView;

		descriptorInfos[2].desc = gpuScrambleImageView;
		{
			ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_INT_OPAQUE_BLACK, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, ISampler::ESMM_NEAREST, 0u, false, ECO_ALWAYS };
			descriptorInfos[2].image.sampler = driver->createGPUSampler(samplerParams);
			descriptorInfos[2].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
		}

		descriptorInfos[3].desc = phiPdfLUTImageView;
		{
			ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, ISampler::ESMM_NEAREST, 0u, false, ECO_ALWAYS };
			descriptorInfos[3].image.sampler = driver->createGPUSampler(samplerParams);
			descriptorInfos[3].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
		}

		descriptorInfos[4].desc = thetaLUTImageView;
		{
			ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, ISampler::ESMM_NEAREST, 0u, false, ECO_ALWAYS };
			descriptorInfos[4].image.sampler = driver->createGPUSampler(samplerParams);
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

		driver->updateDescriptorSets(kDescriptorCount, descriptorSetWrites, 0u, nullptr);
	}

	auto HDRFramebuffer = createHDRFramebuffer(driver, asset::EF_R32G32B32A32_SFLOAT);
	float colorClearValues[] = { 1.f, 1.f, 1.f, 1.f };

	bool ss = true;
	uint64_t lastFPSTime = 0;
	while (device->run() && receiver.keepOpen())
	if (device->isWindowFocused())
	{
		driver->setRenderTarget(HDRFramebuffer, false);
		driver->clearColorBuffer(video::EFAP_COLOR_ATTACHMENT0, colorClearValues);

		camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
		camera->render();

		const auto viewMatrix = camera->getViewMatrix();
		const auto viewProjectionMatrix = camera->getConcatenatedMatrix();
		auto mv = viewMatrix;
		auto mvp = viewProjectionMatrix;
		core::matrix3x4SIMD normalMat;
		mv.getSub3x3InverseTranspose(normalMat);

		SBasicViewParameters uboData;
		memcpy(uboData.MV, mv.pointer(), sizeof(mv));
		memcpy(uboData.MVP, mvp.pointer(), sizeof(mvp));
		memcpy(uboData.NormalMat, normalMat.pointer(), sizeof(normalMat));
		driver->updateBufferRangeViaStagingBuffer(gpuubo.get(), 0ull, sizeof(uboData), &uboData);

		driver->bindGraphicsPipeline(gpuEnvmapPipeline.get());
		driver->bindDescriptorSets(EPBP_GRAPHICS, gpuEnvmapPipeline->getLayout(), 1u, 1u, &uboDescriptorSet1.get(), nullptr);
		driver->bindDescriptorSets(EPBP_GRAPHICS, gpuEnvmapPipeline->getLayout(), 3u, 1u, &descriptorSet5.get(), nullptr);
		driver->drawMeshBuffer(gpuEnvmapMeshBuffer.get());

		driver->setRenderTarget(nullptr, false);
		driver->blitRenderTargets(HDRFramebuffer, nullptr, false, false);

		driver->endScene();

		uint64_t time = device->getTimer()->getRealTime();
		if (time - lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"Envmap Example - Nabla Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str().c_str());
			lastFPSTime = time;
		}

		if (ss)
		{
			ext::ScreenShot::createScreenShot(device, HDRFramebuffer->getAttachment(video::EFAP_COLOR_ATTACHMENT0), "screenshot.exr");
			ss = false;
		}
	}

	const core::vector3df& last_cam_pos = camera->getPosition();
	const core::vectorSIMDf& last_cam_target = camera->getTarget();
	std::cout << "Last camera position: (" << last_cam_pos.X << ", " << last_cam_pos.Y << ", " << last_cam_pos.Z << ")" << std::endl;
	std::cout << "Last camera target: (" << last_cam_target.X << ", " << last_cam_target.Y << ", " << last_cam_target.Z << ")" << std::endl;

	return 0;
}
