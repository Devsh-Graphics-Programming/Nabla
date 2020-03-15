#ifndef _DENOISER_TONEMAPPER_COMMAND_LINE_HANDLER_
#define _DENOISER_TONEMAPPER_COMMAND_LINE_HANDLER_

#include <iostream>
#include <cstdio>
#include <irrlicht.h>

using namespace irr;
using namespace asset;
using namespace core;

#define PROPER_CMD_ARGUMENTS_AMOUNT 7
#define PROPER_BATCH_FILE_ARGUMENTS_AMOUNT 3

enum COMMAND_LINE_MODE
{
	CLM_UNKNOWN,
	CLM_CMD_LIST,
	CLM_FILE,

	CLM_COUNT
};

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

Note there mustn't be any space characters!
Also you mustn't put another data like comments
- behaviour will be undefined, the app'll crash

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

constexpr std::string_view OPENEXR_FILE = "OPENEXR_FILE";
constexpr std::string_view CHANNEL_NAMES = "CHANNEL_NAMES";
constexpr std::string_view EXPOSURE_BIAS = "EXPOSURE_BIAS";
constexpr std::string_view DENOISER_BLEND_FACTOR = "DENOISER_BLEND_FACTOR";
constexpr std::string_view BLOOM_SIZE = "BLOOM_SIZE";
constexpr std::string_view TONEMAPPER = "TONEMAPPER";
constexpr std::string_view REINHARD = "REINHARD";
constexpr std::string_view ACES = "ACES";
constexpr std::string_view OUTPUT = "OUTPUT";

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

using cmdVariableName = std::string;
using rawValuesOfcmdVariable = std::string;
using variablesType = std::array<std::pair<cmdVariableName, core::vector<rawValuesOfcmdVariable>>, DTEA_COUNT>;

class CommandLineHandler
{
	public:

		CommandLineHandler(const int argc, core::vector<std::string> argv, IAssetManager* am);

		auto getVariables() { return &rawVariables; }
		auto getStatus() { return status; }
		auto getMode() { return mode; }

	private:

		bool status;
		COMMAND_LINE_MODE mode;
		variablesType rawVariables;
};

#endif // _DENOISER_TONEMAPPER_COMMAND_LINE_HANDLER_

