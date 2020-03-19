#define _IRR_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <irrlicht.h>

#include "CommandLineHandler.hpp"

using namespace irr;
using namespace asset;
using namespace core;

int main(int argc, char* argv[])
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

	auto getArgvFetchedList = [&]()
	{
		core::vector<std::string> arguments;
		arguments.reserve(PROPER_CMD_ARGUMENTS_AMOUNT);
		for (auto i = 0ul; i < argc; ++i)
			arguments.push_back(argv[i]);

		return arguments;
	};

	auto cmdHandler = CommandLineHandler(argc, getArgvFetchedList(), am);

	if (!cmdHandler.getStatus())
		return 0;

	const auto inputFilesAmount = cmdHandler.getInputFilesAmount();
	const auto fileNamesBundle = cmdHandler.getFileNamesBundle();
	const auto channelNamesBundle = cmdHandler.getChannelNamesBundle();
	const auto cameraTransformBundle = cmdHandler.getCameraTransformBundle();
	const auto exposureBiasBundle = cmdHandler.getExposureBiasBundle();
	const auto denoiserBlendFactorBundle = cmdHandler.getDenoiserBlendFactorBundle();
	const auto bloomSizeBundle = cmdHandler.getBloomSizeBundle();
	const auto tonemapperBundle = cmdHandler.getTonemapperBundle();
	const auto outputFileBundle = cmdHandler.getOutputFileBundle();

	/*
	asset::IAssetLoader::SAssetLoadParams lp;
	auto image_bundle = am->getAsset("../../media/OpenEXR/" + fileName + ".exr", lp);
	assert(!image_bundle.isEmpty());
	auto vertexShader = core::smart_refctd_ptr<ICPUSpecializedShader>();
	auto fragmentShader = core::smart_refctd_ptr<ICPUSpecializedShader>();
	{
		const IAsset::E_TYPE types[]{ IAsset::E_TYPE::ET_SPECIALIZED_SHADER, IAsset::E_TYPE::ET_SPECIALIZED_SHADER, static_cast<IAsset::E_TYPE>(0u) };
		auto bundle = am->findAssets("irr/builtin/materials/debug/uv_debug_shader/specializedshader", types);
		auto refCountedBundle =
		{
			core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(bundle->begin()->getContents().first[0]),
			core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>((bundle->begin() + 1)->getContents().first[0])
		};
		for (auto& shader : refCountedBundle)
		{
			if (shader->getStage() == ISpecializedShader::ESS_VERTEX)
			{
				vertexShader = std::move(shader);
				break;
			}
		}
		fragmentShader = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(core::make_smart_refctd_ptr<asset::ICPUShader>(FRAGMENT_SHADER), asset::ISpecializedShader::SInfo({}, nullptr, "main", ISpecializedShader::E_SHADER_STAGE::ESS_FRAGMENT));
	}
	// TODO pipeline, updating uniforms, drawing, etc
	*/

	return 0;
}