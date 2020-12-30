// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>

using namespace nbl;
using namespace core;
using namespace asset;

int main(int argc, char * argv[])
{
	nbl::SIrrlichtCreationParameters params;
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
	constexpr std::string_view defaultImagePath = "../../media/noises/spp_benchmark_4k_512.exr";
	const auto filePath = std::string(isItDefaultImage ? defaultImagePath.data() : argv[1]);
	auto image_bundle = am->getAsset(filePath, lp);
	assert(!image_bundle.isEmpty());

	for (auto i = 0ul; i < image_bundle.getSize(); ++i)
	{
		ICPUImageView::SCreationParams imgViewParams;
		imgViewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
		imgViewParams.image = IAsset::castDown<ICPUImage>(image_bundle.getContents().begin()[i]);
		imgViewParams.format = imgViewParams.image->getCreationParameters().format;
		imgViewParams.viewType = ICPUImageView::ET_2D;
		imgViewParams.subresourceRange = { static_cast<IImage::E_ASPECT_FLAGS>(0u),0u,1u,0u,1u };
		auto imageView = ICPUImageView::create(std::move(imgViewParams));

		const auto* metadata = static_cast<const COpenEXRImageMetadata*>(imageView->getCreationParameters().image->getMetadata());
		auto channelsName = metadata->getName();

		io::path fileName, extension, finalFileNameWithExtension;
		core::splitFilename(filePath.c_str(), nullptr, &fileName, &extension);
		finalFileNameWithExtension = fileName + ".";
		finalFileNameWithExtension += extension;

		const std::string finalOutputPath = channelsName.empty() ? (std::string(fileName.c_str()) + "_" + std::to_string(i) + "." + std::string(extension.c_str())) : (std::string(fileName.c_str()) + "_" + channelsName + "." + std::string(extension.c_str()));

		const auto params = IAssetWriter::SAssetWriteParams(imageView.get(), EWF_BINARY);
		am->writeAsset(finalOutputPath, params);
	}
		
	return 0;
}
