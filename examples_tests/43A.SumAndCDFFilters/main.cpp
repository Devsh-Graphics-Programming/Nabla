#include <irrlicht.h>

#include "../../../include/irr/asset/filters/CSummedAreaTableImageFilter.h"

using namespace irr;
using namespace core;
using namespace asset;
using namespace video;


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

	auto getSummedImage = [](core::smart_refctd_ptr<ICPUImage> image) -> core::smart_refctd_ptr<ICPUImage>
	{
		using SUM_FILTER = CSummedAreaTableImageFilter<>;

		core::smart_refctd_ptr<ICPUImage> newSumImage;
		{
			auto referenceImageParams = image->getCreationParameters();
			auto referenceBuffer = image->getBuffer();
			auto referenceRegions = image->getRegions();
			auto referenceRegion = referenceRegions.begin();

			auto newImageParams = referenceImageParams;
			auto newCpuBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(referenceBuffer->getSize());
			auto newRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(referenceRegions.size());

			for (auto newRegion = newRegions->begin(); newRegion != newRegions->end(); ++newRegion)
				*newRegion = *(referenceRegion++);

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
			state.scratchMemoryByteSize = state.getRequiredScratchByteSize(referenceImageParams.format, { referenceImageParams.extent.width, referenceImageParams.extent.height, referenceImageParams.extent.depth });
			state.scratchMemory = reinterpret_cast<uint8_t*>(_IRR_ALIGNED_MALLOC(state.scratchMemoryByteSize, 8));

			auto regionWithMipMap = newSumImage->getRegions(0).begin();
			state.extent = regionWithMipMap->getExtent();
			state.layerCount = regionWithMipMap->imageSubresource.layerCount;
			state.inMipLevel = regionWithMipMap->imageSubresource.mipLevel;
			state.outMipLevel = regionWithMipMap->imageSubresource.mipLevel;

			if (!sumFilter.execute(&state))
				os::Printer::log("Something went wrong while performing sum operation!", ELL_WARNING);
		}
		return newSumImage;
	};

	IAssetLoader::SAssetLoadParams lp(0ull, nullptr, IAssetLoader::ECF_DONT_CACHE_REFERENCES);
	auto bundle = assetManager->getAsset("../../media/color_space_test/R8G8B8_1.png", lp);
	auto cpuImage = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(bundle.getContents().first[0]);

	cpuImage = getSummedImage(cpuImage);
}
