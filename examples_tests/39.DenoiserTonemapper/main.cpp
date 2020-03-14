#define _IRR_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <irrlicht.h>

using namespace irr;
using namespace asset;
using namespace core;

const char* FRAGMENT_SHADER = R"(

	vec3 reinhard(vec3 x) 
	{
		return x / (1.0 + x);
	}

	vec3 aces(vec3 x) 
	{
		const float a = 2.51;
		const float b = 0.03;
		const float c = 2.43;
		const float d = 0.59;
		const float e = 0.14;
		return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
	}

	void main()
	{

	}
)";

constexpr std::string_view requiredArgumentsMessage = R"(
Pass appripiate arguments to launch the example or load them using predefined file!

* To load them with a arguments file usage type DenoiserTonemapper.exe -batch <yourargumentfile.txt>
* To load them passing arguments through cmd.

Loading syntax:

-OPENEXR_FILE=filename
-CHANNEL_NAMES=name,name,name,... 
-EXPOSURE_BIAS=value
-DENOISER_BLEND_FACTOR=value
-BLOOM_SIZE=x,y
-TONEMAPPER=tonemapper=arg1,arg2,arg3,...
-OUTPUT=file.choosenextension

Description and usage:

OPENEXR_FILE: OpenEXR file containing various channels - type without extension
CHANNEL_NAMES: name of denoiser input channels - split each next channel using ","
EXPOSURE_BIAS: exposure bias value used in shader
DENOISER_BLEND_FACTOR: denoiser blend factor used in shader
BLOOM_SIZE: bloom size
TONEMAPPER: tonemapper - choose between "REINHARD" and "ACES". After specifing it
you have to assing arguments to revelant tonemapper. For "REINHARD" tonemapper
there are no arguments, so you should not type anything else, but for "ACES"
you have to specify some arguments for it's curve function. They are following:

arg1=value
arg2=value
arg3=value
arg4=value
arg5=value

where function is:
f(x) = clamp((x * (arg1 * x + arg2)) / (x * (arg3 * x + arg4) + arg5), 0.0, 1.0)

so for example, specifing "REINHARD" tonemapper looks like:
-TONEMAPPER=REINHARD

and specifing "ACES" looks like:
-TONEMAPPER=ACES=arg1,arg2,arg3,arg4,arg5

OUTPUT: output file with specified extension 

)";

auto constexpr PROPER_ARGUMENTS_AMOUNT = 7;

enum DENOISER_TONEMAPPER_EXAMPLE_ARGUMENTS
{
	DTEA_OPENEXR_FILE,
	DTEA_CHANNEL_NAMES,
	DTEA_EXPOSURE_BIAS,
	DTEA_DENOISER_BLEND_FACTOR,
	DTEA_BLOOM_SIZE,
	DTEA_REINHARD,
	DTEA_ACES,
	DTEA_OUTPUT,

	DTEA_COUNT
};

enum ACES_ARGUMENTS
{
	AA_ARG1,
	AA_ARG2,
	AA_ARG3,
	AA_ARG4,
	AA_ARG5,

	AA_COUNT
};

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

	using cmdVariableName = std::string;
	using rawValuesOfcmdVariable = std::string;
	auto createUnfilledRawVariableMapper = [&]()
	{
		std::array<std::pair<cmdVariableName, core::vector<rawValuesOfcmdVariable>>, DTEA_COUNT> variables;
		variables[DTEA_OPENEXR_FILE].first = "OPENEXR_FILE";
		variables[DTEA_CHANNEL_NAMES].first = "CHANNEL_NAMES";
		variables[DTEA_EXPOSURE_BIAS].first = "EXPOSURE_BIAS";
		variables[DTEA_DENOISER_BLEND_FACTOR].first = "DENOISER_BLEND_FACTOR";
		variables[DTEA_BLOOM_SIZE].first = "BLOOM_SIZE";
		variables[DTEA_REINHARD].first = "REINHARD";
		variables[DTEA_ACES].first = "ACES";
		variables[DTEA_OUTPUT].first = "OUTPUT";
		return variables;
	};

	 auto rawVariables = createUnfilledRawVariableMapper();

	if (argc == 1)
		os::Printer::log(requiredArgumentsMessage.data(), ELL_ERROR);
	else if (argc == 7)
	{
		for (auto i = 0; i < DTEA_COUNT; ++i)
		{
			auto referenceVariableMap = rawVariables[i];

			for (auto z = 0; z < PROPER_ARGUMENTS_AMOUNT; ++z)
			{
				std::string rawFetchedCmdArgument = argv[z];
				const auto offset = rawFetchedCmdArgument.find_last_of("-") + 1;
				const auto endOfFetchedVariableName = rawFetchedCmdArgument.find_last_of(",") + 1;
				const auto count = endOfFetchedVariableName - offset;
				const auto cmdFetchedVariable = rawFetchedCmdArgument.substr(offset, count);

				auto getSerializedValues = [&](auto variablesStream, auto supposedArgumentsAmout = 1) 
				{
					core::vector<std::string> variablesHandle;
					variablesHandle.reserve(supposedArgumentsAmout);

					std::string tmpStream;
					for (auto x = 0ul; x < variablesStream.size(); ++x)
					{
						const auto character = variablesStream.at(x);

						if (character == '\0' || character == ',')
						{
							variablesHandle.push_back(tmpStream);
							tmpStream.clear();
						}
						else
							tmpStream.push_back(character);
					}

					return variablesHandle;
				};

				const bool matchedVariables = referenceVariableMap.first == cmdFetchedVariable;
				if (matchedVariables)
				{
					const auto beginningOfVariables = rawFetchedCmdArgument.find_last_of("=") + 1;
					auto variablesStream = rawFetchedCmdArgument.substr(beginningOfVariables);

					if (cmdFetchedVariable == "ACES")
					{
						auto variablesHandle = getSerializedValues(variablesStream, AA_COUNT); // 5 values
						
						// if there is no ACES argument - defaults are assumed
						referenceVariableMap.second[AA_ARG1] = variablesHandle[AA_ARG1].empty() ? std::string("2.51") : variablesHandle[AA_ARG1];
						referenceVariableMap.second[AA_ARG2] = variablesHandle[AA_ARG2].empty() ? std::string("0.03") : variablesHandle[AA_ARG2];
						referenceVariableMap.second[AA_ARG3] = variablesHandle[AA_ARG3].empty() ? std::string("2.43") : variablesHandle[AA_ARG3];
						referenceVariableMap.second[AA_ARG4] = variablesHandle[AA_ARG4].empty() ? std::string("0.59") : variablesHandle[AA_ARG4];
						referenceVariableMap.second[AA_ARG5] = variablesHandle[AA_ARG5].empty() ? std::string("0.14") : variablesHandle[AA_ARG5];
					}
					else if (cmdFetchedVariable == "CHANNEL_NAMES") // various amount of values allowed
					{
						auto variablesHandle = getSerializedValues(variablesStream, 3); // the number is only for allocation optimalization, but the assumption is 3 for color, albedo and normal channel bundle
						for (auto b = 0; b < variablesHandle.size(); ++b)
							referenceVariableMap.second[b] = variablesHandle[b];
					}
					else // always one value 
					{
						auto variablesHandle = getSerializedValues(variablesStream, 1);
						referenceVariableMap.second[0] = variablesHandle[0];
					}
				}
				else
				{
					continue; // search further
				}
					
			}
		}
	}
	else if (argc > 1 && argc < 7)
	{
		os::Printer::log(std::string("No assumptions allowed yet - too less arguments!") + requiredArgumentsMessage.data(), ELL_ERROR);
		return 0;
	}
	else if (argc > 7)
	{
		os::Printer::log(std::string("Too many arguments!") + requiredArgumentsMessage.data(), ELL_ERROR);
		return 0;
	}

	auto driver = device->getVideoDriver();
	auto smgr = device->getSceneManager();
	auto am = device->getAssetManager();

	const auto filename = rawVariables[DTEA_OPENEXR_FILE].second[0];
	asset::IAssetLoader::SAssetLoadParams lp;
	auto image_bundle = am->getAsset("../../media/OpenEXR/" + filename + ".exr", lp);
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
		
	return 0;
}
