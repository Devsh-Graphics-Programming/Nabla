#include <irrlicht.h>

#include "../../../include/irr/asset/filters/CSummedAreaTableImageFilter.h"
#include "../ext/ScreenShot/ScreenShot.h"

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
			newImageParams.format = EF_R16G16B16A16_UNORM;
			const float texelByteSizeFactor = asset::getTexelOrBlockBytesize(newImageParams.format) / asset::getTexelOrBlockBytesize(referenceImageParams.format);
			auto newCpuBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(referenceBuffer->getSize() * texelByteSizeFactor);
			auto newRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(referenceRegions.size());

			for (auto newRegion = newRegions->begin(); newRegion != newRegions->end(); ++newRegion)
			{
				*newRegion = *(referenceRegion++);
				newRegion->bufferOffset *= texelByteSizeFactor;
			}

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
			state.scratchMemoryByteSize = state.getRequiredScratchByteSize(state.inImage);
			state.scratchMemory = reinterpret_cast<uint8_t*>(_IRR_ALIGNED_MALLOC(state.scratchMemoryByteSize, 8));

			auto stateExtent = newSumImage->getMipSize(0);
			state.extent = { stateExtent.X, stateExtent.Y, stateExtent.Z };
			state.layerCount = newSumImage->getCreationParameters().arrayLayers;
			state.inMipLevel = 0;
			state.outMipLevel = 0;

			if (!sumFilter.execute(&state))
				os::Printer::log("Something went wrong while performing sum operation!", ELL_WARNING);

			_IRR_ALIGNED_FREE(state.scratchMemory);
		}
		return newSumImage;
	};

	IAssetLoader::SAssetLoadParams lp(0ull, nullptr, IAssetLoader::ECF_DONT_CACHE_REFERENCES);
	//auto bundle = assetManager->getAsset("../../media/colorexr.exr", lp);
	auto bundle = assetManager->getAsset("../../media/GLI/earth-cubemap2.dds", lp);
	auto cpuImageViewFetched = core::smart_refctd_ptr_static_cast<asset::ICPUImageView>(bundle.getContents().first[0]);

	auto cpuImage = getSummedImage(cpuImageViewFetched->getCreationParameters().image);

	ICPUImageView::SCreationParams viewParams;
	viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
	viewParams.image = cpuImage;
	viewParams.format = viewParams.image->getCreationParameters().format;
	viewParams.viewType = IImageView<ICPUImage>::ET_2D;
	viewParams.subresourceRange.baseArrayLayer = 0u;
	viewParams.subresourceRange.layerCount = 1u;
	viewParams.subresourceRange.baseMipLevel = 0u;
	viewParams.subresourceRange.levelCount = 1u;

	auto cpuImageView = ICPUImageView::create(std::move(viewParams));

	asset::IAssetWriter::SAssetWriteParams wparams(cpuImageView.get());
	assetManager->writeAsset("SAT_OUTPUTdds.dds", wparams);
}
