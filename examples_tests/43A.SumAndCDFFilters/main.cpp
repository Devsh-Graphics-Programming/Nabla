#include <irrlicht.h>

#include "../../../include/irr/asset/filters/CSummedAreaTableImageFilter.h"
#include "../ext/ScreenShot/ScreenShot.h"

using namespace irr;
using namespace core;
using namespace asset;
using namespace video;

/*
	Comment IMAGE_VIEW define to use ordinary cpu image.
	You can view the results in Renderdoc.
*/

#define IMAGE_VIEW 

int main()
{
	irr::SIrrlichtCreationParameters params;
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
		using SUM_FILTER = CSummedAreaTableImageFilter<false>;

		core::smart_refctd_ptr<ICPUImage> newSumImage;
		{
			const auto referenceImageParams = image->getCreationParameters();
			const auto referenceBuffer = image->getBuffer();
			const auto referenceRegions = image->getRegions();
			const auto* referenceRegion = referenceRegions.begin();

			auto newImageParams = referenceImageParams;

			#ifdef IMAGE_VIEW
			newImageParams.flags = IImage::ECF_CUBE_COMPATIBLE_BIT;
			newImageParams.format = EF_R32G32B32A32_SFLOAT;
			#else
			newImageParams.format = EF_R16G16B16A16_UNORM;
			#endif // IMAGE_VIEW

			auto newRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(referenceRegions.size());
			size_t regionOffsets = {};

			for (auto newRegion = newRegions->begin(); newRegion != newRegions->end(); ++newRegion)
			{
				*newRegion = *(referenceRegion++);
				newRegion->bufferOffset = regionOffsets;
				regionOffsets += newRegion->imageExtent.width * newRegion->imageExtent.height * newRegion->imageExtent.depth * newRegion->imageSubresource.layerCount * asset::getTexelOrBlockBytesize(newImageParams.format);
			}

			auto newCpuBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(regionOffsets);
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
			state.scratchMemoryByteSize = state.getRequiredScratchByteSize(state.inImage, state.outImage);
			state.scratchMemory = reinterpret_cast<uint8_t*>(_IRR_ALIGNED_MALLOC(state.scratchMemoryByteSize, 8));
			
			state.extent = { referenceImageParams.extent.width, referenceImageParams.extent.height, referenceImageParams.extent.depth };
			state.layerCount = newSumImage->getCreationParameters().arrayLayers;

			#ifdef IMAGE_VIEW
			state.inMipLevel = 2;
			state.outMipLevel = 2;
			state.normalizeImageByTotalSATValues = true;
			#else
			state.inMipLevel = 0;
			state.outMipLevel = 0;
			#endif // IMAGE_VIEW

			if (!sumFilter.execute(&state))
				os::Printer::log("Something went wrong while performing sum operation!", ELL_WARNING);

			_IRR_ALIGNED_FREE(state.scratchMemory);
		}
		return newSumImage;
	};

	IAssetLoader::SAssetLoadParams lp(0ull, nullptr, IAssetLoader::ECF_DONT_CACHE_REFERENCES);

	#ifdef IMAGE_VIEW
	auto bundle = assetManager->getAsset("../../media/GLI/earth-cubemap3.dds", lp);
	auto cpuImageViewFetched = core::smart_refctd_ptr_static_cast<asset::ICPUImageView>(bundle.getContents().first[0]);

	auto cpuImage = getSummedImage(cpuImageViewFetched->getCreationParameters().image);
	#else
	auto bundle = assetManager->getAsset("../../media/colorexr.exr", lp);
	auto cpuImage = getSummedImage(core::smart_refctd_ptr_static_cast<asset::ICPUImage>(bundle.getContents().first[0]));
	#endif // IMAGE_VIEW

	ICPUImageView::SCreationParams viewParams;
	viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
	viewParams.image = cpuImage;
	viewParams.format = viewParams.image->getCreationParameters().format;

	#ifdef IMAGE_VIEW
	viewParams.viewType = IImageView<ICPUImage>::ET_2D_ARRAY;
	#else
	viewParams.viewType = IImageView<ICPUImage>::ET_2D;
	#endif // IMAGE_VIEW

	viewParams.subresourceRange.baseArrayLayer = 0u;
	viewParams.subresourceRange.layerCount = cpuImage->getCreationParameters().arrayLayers;
	viewParams.subresourceRange.baseMipLevel = 0u;
	viewParams.subresourceRange.levelCount = cpuImage->getCreationParameters().mipLevels;

	auto cpuImageView = ICPUImageView::create(std::move(viewParams));
	assert(cpuImageView.get(), "The imageView didn't passed creation validation!");

	asset::IAssetWriter::SAssetWriteParams wparams(cpuImageView.get());
	#ifdef IMAGE_VIEW
	assetManager->writeAsset("SAT_OUTPUT.dds", wparams);
	#else
	assetManager->writeAsset("SAT_OUTPUT.exr", wparams);
	#endif // IMAGE_VIEW
}
