// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <nabla.h>

#include "../../../include/nbl/asset/filters/CSummedAreaTableImageFilter.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"

using namespace nbl;
using namespace core;
using namespace asset;
using namespace video;
zzz
/*
	Comment IMAGE_VIEW define to use ordinary cpu image.
	You can view the results in Renderdoc.

	When using ordinary IMAGE you can use OVERLAPPING_REGIONS
	to choose whether to use extra overlapping region on output image
	with a custom in-offset and extent

	You can also specify whether to perform sum in
	exclusive mode by EXCLUSIVE_SUM,  
	otherwise in inclusive mode 
*/

// #define IMAGE_VIEW 
// #define OVERLAPPING_REGIONS			
constexpr bool EXCLUSIVE_SUM = true;
constexpr auto MIPMAP_IMAGE_VIEW = 2u;		// feel free to change the mipmap
constexpr auto MIPMAP_IMAGE = 0u;			// ordinary image used in the example has only 0-th mipmap

/*
	Discrete convolution for getting input image after SAT calculations

	- support [-1.5,1.5]
	- (weight = -1) in [-1.5,-0.5]
	- (weight = 1) in [-0.5,0.5]
	- (weight = 0) in [0.5,1.5] and in range over the support
*/

using CDiscreteConvolutionRatioForSupport = std::ratio<3, 2>; //!< 1.5
class CDiscreteConvolutionFilterKernel : public CFloatingPointSeparableImageFilterKernelBase<CDiscreteConvolutionFilterKernel>
{
		using Base = CFloatingPointSeparableImageFilterKernelBase<CDiscreteConvolutionFilterKernel>;

	public:
		CDiscreteConvolutionFilterKernel() : Base(1.5f,0.5f) {}

		inline float weight(float x, int32_t channel) const
		{
			if (x >= -1.5f && x <= -0.5f)
				return -1.0f;
			else if (x >= -0.5f && x <= 0.5f)
				return 1.0f;
			else
				return 0.0f;
		}
};

int main()
{
	nbl::SIrrlichtCreationParameters params;
	params.Bits = 32;
	params.ZBufferBits = 24;
	params.DriverType = video::EDT_OPENGL;
	params.WindowSize = dimension2d<uint32_t>(1600, 900);
	params.Fullscreen = false;
	params.Doublebuffer = true;
	params.Vsync = true;
	params.Stencilbuffer = false;

	auto device = createDeviceEx(params);
	if (!device)
		return false;

	device->getCursorControl()->setVisible(false);
	auto driver = device->getVideoDriver();
	auto assetManager = device->getAssetManager();
	auto sceneManager = device->getSceneManager();

	auto getSummedImage = [](const core::smart_refctd_ptr<ICPUImage> image) -> core::smart_refctd_ptr<ICPUImage>
	{
		using SUM_FILTER = CSummedAreaTableImageFilter<EXCLUSIVE_SUM>;

		core::smart_refctd_ptr<ICPUImage> newSumImage;
		{
			const auto referenceImageParams = image->getCreationParameters();
			const auto referenceBuffer = image->getBuffer();
			const auto referenceRegions = image->getRegions();
			const auto* referenceRegion = referenceRegions.begin();

			auto newImageParams = referenceImageParams;
			core::smart_refctd_ptr<ICPUBuffer> newCpuBuffer;

			#ifdef IMAGE_VIEW
			newImageParams.flags = IImage::ECF_CUBE_COMPATIBLE_BIT;
			newImageParams.format = EF_R16G16B16A16_UNORM;
			#else
			newImageParams.format = EF_R32G32B32A32_SFLOAT;
			#endif // IMAGE_VIEW

			auto newRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>
			(
				#ifdef IMAGE_VIEW
				referenceRegions.size()
				#else

				#ifdef OVERLAPPING_REGIONS
				2u
				#else
				referenceRegions.size() // one region at all
				#endif // OVERLAPPING_REGIONS

				#endif // IMAGE_VIEW
			);

			size_t regionOffsets = {};

			#ifdef IMAGE_VIEW
			for (auto newRegion = newRegions->begin(); newRegion != newRegions->end(); ++newRegion)
			{
				/*
					Regions pulled directly from a loader doesn't overlap, so each following is a certain single mipmap
				*/

				auto idOffset = newRegion - newRegions->begin();
				*newRegion = *(referenceRegion++);
				newRegion->bufferOffset = regionOffsets;

				const auto fullMipMapExtent = image->getMipSize(idOffset);

				regionOffsets += fullMipMapExtent.x * fullMipMapExtent.y * fullMipMapExtent.z * newImageParams.arrayLayers * asset::getTexelOrBlockBytesize(newImageParams.format);
			}
			newCpuBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(regionOffsets);
			#else

			/*
				2 overlapping regions if OVERLAPPING_REGIONS is defined
			*/

			const auto fullMipMapExtent = image->getMipSize(MIPMAP_IMAGE);
			const auto info = image->getTexelBlockInfo();
			const auto fullMipMapExtentInBlocks = info.convertTexelsToBlocks(fullMipMapExtent);
			const size_t bufferByteSize = fullMipMapExtentInBlocks.x * fullMipMapExtentInBlocks.y * fullMipMapExtentInBlocks.z * newImageParams.arrayLayers * asset::getTexelOrBlockBytesize(newImageParams.format);
			newCpuBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(bufferByteSize);

			auto newFirstRegion = newRegions->begin();
			*newFirstRegion = *(referenceRegion++);
			newFirstRegion->bufferOffset = regionOffsets;

			#ifdef OVERLAPPING_REGIONS
			auto newSecondRegion = newRegions->begin() + 1;
			*newSecondRegion = *newFirstRegion;

			newSecondRegion->bufferRowLength = fullMipMapExtent.x;
			newSecondRegion->bufferImageHeight = fullMipMapExtent.y;

			auto simdImageOffset = fullMipMapExtent / 4;
			newSecondRegion->imageOffset = { simdImageOffset.x, simdImageOffset.y, simdImageOffset.z };

			auto simdImageExtent = fullMipMapExtent / 2;
			newSecondRegion->imageExtent = { simdImageExtent.x, simdImageExtent.y, 1 };

			newSecondRegion->bufferOffset = newFirstRegion->getByteOffset(simdImageOffset,newFirstRegion->getByteStrides(TexelBlockInfo(newImageParams.format)));
			#endif // OVERLAPPING_REGIONS

			#endif // IMAGE_VIEW

			newSumImage = ICPUImage::create(std::move(newImageParams));
			newSumImage->setBufferAndRegions(std::move(newCpuBuffer), newRegions);

			SUM_FILTER sumFilter;
			SUM_FILTER::state_type state;
			
			state.inImage = image.get();
			state.outImage = newSumImage.get();
			state.inOffset = { 0, 0, 0 };
			state.inBaseLayer = 0;
			state.outOffset = { 0, 0, 0 };
			state.outBaseLayer = 0;

			#ifdef IMAGE_VIEW
			const auto fullMipMapExtent = image->getMipSize(MIPMAP_IMAGE_VIEW);
			state.extent = { fullMipMapExtent.x, fullMipMapExtent.y, fullMipMapExtent.z };
			#else 
			state.extent =  { referenceImageParams.extent.width, referenceImageParams.extent.height, referenceImageParams.extent.depth };
			#endif // IMAGE_VIEW

			state.layerCount = newSumImage->getCreationParameters().arrayLayers;
			
			state.scratchMemoryByteSize = state.getRequiredScratchByteSize(state.inImage, state.extent);
			state.scratchMemory = reinterpret_cast<uint8_t*>(_NBL_ALIGNED_MALLOC(state.scratchMemoryByteSize, 32));
			#ifdef IMAGE_VIEW
			state.inMipLevel = MIPMAP_IMAGE_VIEW;
			state.outMipLevel = MIPMAP_IMAGE_VIEW;
			state.normalizeImageByTotalSATValues = true; // pay attention that we may force normalizing output values (but it will do it anyway if input is normalized)
			#else
			state.inMipLevel = MIPMAP_IMAGE;
			state.outMipLevel = MIPMAP_IMAGE;
			#endif // IMAGE_VIEW

			if (!sumFilter.execute(&state))
				os::Printer::log("Something went wrong while performing sum operation!", ELL_WARNING);

			_NBL_ALIGNED_FREE(state.scratchMemory);
		}
		return newSumImage;
	};

	IAssetLoader::SAssetLoadParams lp(0ull, nullptr, IAssetLoader::ECF_DONT_CACHE_REFERENCES);

	#ifdef IMAGE_VIEW
	auto bundle = assetManager->getAsset("../../media/GLI/earth-cubemap3.dds", lp);
	auto cpuImageViewFetched = core::smart_refctd_ptr_static_cast<asset::ICPUImageView>(bundle.getContents().begin()[0]);

	auto cpuImage = getSummedImage(cpuImageViewFetched->getCreationParameters().image);
	#else
	auto bundle = assetManager->getAsset("../../media/colorexr.exr", lp);
	auto cpuImage = getSummedImage(core::smart_refctd_ptr_static_cast<asset::ICPUImage>(bundle.getContents().begin()[0]));
	#endif // IMAGE_VIEW

	ICPUImageView::SCreationParams viewParams;
	viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
	viewParams.image = cpuImage;
	viewParams.format = viewParams.image->getCreationParameters().format;

	#ifdef IMAGE_VIEW
	viewParams.components = cpuImageViewFetched->getComponents();
	viewParams.viewType = IImageView<ICPUImage>::ET_2D_ARRAY;
	#else
	viewParams.viewType = IImageView<ICPUImage>::ET_2D;
	#endif // IMAGE_VIEW

	viewParams.subresourceRange.baseArrayLayer = 0u;
	viewParams.subresourceRange.layerCount = cpuImage->getCreationParameters().arrayLayers;
	viewParams.subresourceRange.baseMipLevel = 0u;
	viewParams.subresourceRange.levelCount = cpuImage->getCreationParameters().mipLevels;

	auto cpuImageView = ICPUImageView::create(std::move(viewParams));
	assert(cpuImageView.get(), "The imageView didn't pass creation validation!");

	auto writeSATandGetItsOutputName = [&]() -> std::string
	{
		std::string outputFileName;

		constexpr std::string_view MODE = [&]() constexpr
		{
			if constexpr (EXCLUSIVE_SUM)
				return "EXCLUSIVE_SAT_";
			else
				return "INCLUSIVE_SAT_";
		}
		();

		asset::IAssetWriter::SAssetWriteParams wparams(cpuImageView.get());
		#ifdef IMAGE_VIEW
		assetManager->writeAsset(outputFileName = std::string(MODE.data()) + "IMG_VIEW.dds", wparams);
		#else
			#ifdef OVERLAPPING_REGIONS
			assetManager->writeAsset(outputFileName = std::string(MODE.data()) + "IMG_OVERLAPPING_REGIONS.exr", wparams);
			#else 
			assetManager->writeAsset(outputFileName = std::string(MODE.data()) + "IMG.exr", wparams);
			#endif // OVERLAPPING_REGIONS
		#endif // IMAGE_VIEW

		return outputFileName;
	};

	auto getDisConvolutedImage = [&](const core::smart_refctd_ptr<ICPUImage> inImage) -> core::smart_refctd_ptr<ICPUImage>
	{
		auto outImage = core::move_and_static_cast<ICPUImage>(inImage->clone());

		using DISCRETE_CONVOLUTION_BLIT_FILTER = asset::CBlitImageFilter<false,true,DefaultSwizzle,CWhiteNoiseDither,CDiscreteConvolutionFilterKernel,CDiscreteConvolutionFilterKernel,CBoxImageFilterKernel>;
		DISCRETE_CONVOLUTION_BLIT_FILTER blitImageFilter;
		DISCRETE_CONVOLUTION_BLIT_FILTER::state_type state;
		
		core::vectorSIMDu32 extentLayerCount;
		#ifdef IMAGE_VIEW
		state.inMipLevel = MIPMAP_IMAGE_VIEW;
		state.outMipLevel = MIPMAP_IMAGE_VIEW;
		extentLayerCount = core::vectorSIMDu32(0, 0, 0, inImage->getCreationParameters().arrayLayers) + inImage->getMipSize(MIPMAP_IMAGE_VIEW);
		#else
		state.inMipLevel = MIPMAP_IMAGE;
		state.outMipLevel = MIPMAP_IMAGE;
		extentLayerCount = core::vectorSIMDu32(0, 0, 0, inImage->getCreationParameters().arrayLayers) + inImage->getMipSize(MIPMAP_IMAGE);
		#endif // IMAGE_VIEW

		state.inOffsetBaseLayer = core::vectorSIMDu32();
		state.inExtentLayerCount = extentLayerCount;
		state.inImage = inImage.get();

		state.outOffsetBaseLayer = core::vectorSIMDu32();
		state.outExtentLayerCount = extentLayerCount;
		state.outImage = outImage.get();

		state.swizzle = {};

		state.ditherState = _NBL_NEW(std::remove_pointer<decltype(state.ditherState)>::type);
		state.scratchMemoryByteSize = blitImageFilter.getRequiredScratchByteSize(&state);
		state.scratchMemory = reinterpret_cast<uint8_t*>(_NBL_ALIGNED_MALLOC(state.scratchMemoryByteSize, 32));

		if (!blitImageFilter.execute(&state))
			os::Printer::log("Something went wrong while performing discrete convolution operation!", ELL_WARNING);

		_NBL_DELETE(state.ditherState);
		_NBL_ALIGNED_FREE(state.scratchMemory);

		return outImage;
	};

	{
		const std::string satFileName = writeSATandGetItsOutputName();
		const std::string convolutedSatFileName = "CONVOLUTED_" + satFileName;

		auto bundle = assetManager->getAsset(satFileName, lp);
		#ifdef IMAGE_VIEW
		auto cpuImageViewFetched = core::smart_refctd_ptr_static_cast<asset::ICPUImageView>(bundle.getContents().begin()[0]);
		auto cpuImage = getDisConvolutedImage(cpuImageViewFetched->getCreationParameters().image);
		#else
		auto cpuImage = getDisConvolutedImage(core::smart_refctd_ptr_static_cast<asset::ICPUImage>(bundle.getContents().begin()[0]));
		#endif // IMAGE_VIEW

		viewParams.image = cpuImage;
		viewParams.format = cpuImage->getCreationParameters().format;
		viewParams.components = {};

		auto cpuImageView = ICPUImageView::create(std::move(viewParams));
		assert(cpuImageView.get(), "The imageView didn't pass creation validation!");

		asset::IAssetWriter::SAssetWriteParams wparams(cpuImageView.get());
		assetManager->writeAsset(convolutedSatFileName, wparams);
	}
}
