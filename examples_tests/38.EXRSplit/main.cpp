#define _IRR_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <irrlicht.h>

using namespace irr;
using namespace core;
using namespace asset;

int main(int argc, char * argv[])
{
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; 
	params.ZBufferBits = 24; 
	params.DriverType = video::EDT_NULL; 
	params.WindowSize = dimension2d<uint32_t>(1280, 720);
	params.Fullscreen = false;
	params.Vsync = true; 
	params.Doublebuffer = true;
	params.Stencilbuffer = false; 
	auto device = createDeviceEx(params);

	if (!device)
		return 1;

	const bool isItDefaultImage = argc == 1;
	if (isItDefaultImage)
		os::Printer::log("No image specified - loading a default OpenEXR image placed in media/OpenEXR!", ELL_INFORMATION);
	else if (argc == 2)
		os::Printer::log(argv[1] + std::string(" specified!"), ELL_INFORMATION);
	else
	{
		os::Printer::log("To many arguments - pass a single filename without .exr extension of OpenEXR image placed in media/OpenEXR! ", ELL_ERROR);
		return 0;
	}

	auto driver = device->getVideoDriver();
	auto smgr = device->getSceneManager();
	auto am = device->getAssetManager();

	IAssetLoader::SAssetLoadParams lp;
	auto image_bundle = am->getAsset(std::string(isItDefaultImage ? "../../media/OpenEXR/daily_pt_16.exr" : argv[1]), lp);
	assert(!image_bundle.isEmpty());

	for (auto i = 0ul; i < image_bundle.getSize(); ++i)
	{
		ICPUImageView::SCreationParams imgViewParams;
		imgViewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
		imgViewParams.image = IAsset::castDown<ICPUImage>(image_bundle.getContents().first[i]);
		imgViewParams.format = imgViewParams.image->getCreationParameters().format;
		imgViewParams.viewType = ICPUImageView::ET_2D;
		imgViewParams.subresourceRange = { static_cast<IImage::E_ASPECT_FLAGS>(0u),0u,1u,0u,1u };
		auto imageView = ICPUImageView::create(std::move(imgViewParams));

		const auto params = IAssetWriter::SAssetWriteParams(imageView.get(), EWF_BINARY);
		am->writeAsset("OpenEXR_" + std::to_string(i) + ".exr", params);
	}
		
	return 0;
}
