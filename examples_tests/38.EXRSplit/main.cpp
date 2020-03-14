#define _IRR_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <irrlicht.h>

using namespace irr;
using namespace core;

int main(int argc, char * argv[])
{
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; 
	params.ZBufferBits = 24; 
	params.DriverType = video::EDT_OPENGL; 
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
		os::Printer::log(argv[1] + std::string(".exr specified!"), ELL_INFORMATION);
	else
	{
		os::Printer::log("To many arguments - pass a single filename without .exr extension of OpenEXR image placed in media/OpenEXR! ", ELL_ERROR);
		return 0;
	}

	auto driver = device->getVideoDriver();
	auto smgr = device->getSceneManager();
	auto am = device->getAssetManager();

	asset::IAssetLoader::SAssetLoadParams lp;
	auto image_bundle = am->getAsset("../../media/OpenEXR/" + std::string(isItDefaultImage ? "daily_pt_1" : argv[1]) + ".exr", lp);
	assert(!image_bundle.isEmpty());

	for (auto i = 0ul; i < image_bundle.getSize(); ++i)
	{
		auto image = image_bundle.getContents().first[i];
		const auto params = asset::IAssetWriter::SAssetWriteParams(image.get(), asset::EWF_BINARY);
		am->writeAsset("OpenEXR_" + std::to_string(i) + ".exr", params);
	}
		
	return 0;
}
