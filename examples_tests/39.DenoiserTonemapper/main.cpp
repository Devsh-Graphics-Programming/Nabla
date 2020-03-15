#define _IRR_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <irrlicht.h>

#include "CommandLineHandler.hpp"

using namespace irr;
using namespace asset;
using namespace core;

// TODO
const char* FRAGMENT_SHADER = R"(

	vec3 reinhard(vec3 x) 
	{
		return x / (1.0 + x);
	}

	vec3 aces(vec3 x, float arg1, float arg2, float arg3, float arg4, float arg5) 
	{
		return clamp((x * (arg1 * x + arg2)) / (x * (arg3 * x + arg4) + arg5), 0.0, 1.0);
	}

	void main()
	{
		
	}
)";

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

	auto getTonemapper = [&](const variablesType& variables) -> std::pair<std::string, core::vector<float>>
	{
		core::vector<float> values(AA_COUNT);
		auto tonemapper = variables[DTEA_ACES].second.empty() ? variables[DTEA_REINHARD] : variables[DTEA_ACES];

		for (auto i = 0; i < values.size(); ++i)
			*(values.begin() + i) = std::stof(tonemapper.second[i]);

		return { tonemapper.first, values };
	};

	const auto variables = *cmdHandler.getVariables();
	const auto fileName = variables[DTEA_OPENEXR_FILE].second[0];
	const auto channelNames = variables[DTEA_CHANNEL_NAMES].second;
	const auto exposureBias = std::stof(variables[DTEA_EXPOSURE_BIAS].second[0]);
	const auto denoiserBlendFactor = std::stof(variables[DTEA_DENOISER_BLEND_FACTOR].second[0]);
	const auto bloomSize = core::vector2df(std::stof(variables[DTEA_BLOOM_SIZE].second[0]), std::stof(variables[DTEA_BLOOM_SIZE].second[1]));
	const auto tonemapper = getTonemapper(variables);
	const auto outputFile = variables[DTEA_OUTPUT].second[0];
	
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
