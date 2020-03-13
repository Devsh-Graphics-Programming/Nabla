#define _IRR_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <irrlicht.h>

using namespace irr;
using namespace core;

int main()
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

	auto driver = device->getVideoDriver();
	auto smgr = device->getSceneManager();
	auto am = device->getAssetManager();

	asset::IAssetLoader::SAssetLoadParams lp;
	auto image_bundle = am->getAsset("../../media/daily_pt_1.exr", lp);
	assert(!image_bundle.isEmpty());

	for (auto i = 0ul; i < image_bundle.getSize(); ++i)
	{
		auto image = image_bundle.getContents().first[i];
		const auto params = asset::IAssetWriter::SAssetWriteParams(image.get(), asset::EWF_BINARY);
		am->writeAsset("OpenEXR_" + std::to_string(i) + ".exr", params);
	}
		
	return 0;
}
