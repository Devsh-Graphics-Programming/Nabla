#ifndef _DENOISER_TONEMAPPER_COMMAND_LINE_HANDLER_
#define _DENOISER_TONEMAPPER_COMMAND_LINE_HANDLER_

#include <iostream>
#include <cstdio>
#include <chrono>
#include "irrlicht.h"
#include "irr/core/core.h"
#include "../../ext/MitsubaLoader/CMitsubaLoader.h"

#define PROPER_CMD_ARGUMENTS_AMOUNT 14
#define MANDATORY_CMD_ARGUMENTS_AMOUNT 8
#define OPTIONAL_CMD_ARGUMENTS_AMOUNT 6
#define PROPER_BATCH_FILE_ARGUMENTS_AMOUNT 3

enum COMMAND_LINE_MODE
{
	CLM_UNKNOWN,
	CLM_CMD_LIST,
	CLM_BATCH_INPUT,

	CLM_COUNT
};

constexpr std::string_view requiredArgumentsMessage = R"(
Pass appripiate arguments to launch the example or load them using predefined file!
* To load them with a arguments file usage type DenoiserTonemapper.exe -batch <yourargumentfile.txt>
* To load them passing arguments through cmd.

Mandatory parameters:
-COLOR_FILE=colorFilePath
-CAMERA_TRANSFORM=mitsubaFilePath or val1,val2,val3,...,val9
-MEDIAN_FILTER_RADIUS=value
-DENOISER_EXPOSURE_BIAS=value
-DENOISER_BLEND_FACTOR=value
-BLOOM_FOV=theta
-TONEMAPPER=tonemapper=keyValue,extraParameter
-OUTPUT=file.choosenextension
Optional Parameters:
-ALBEDO_FILE=albedoFilePath
-NORMAL_FILE=normalFilePath
-COLOR_CHANNEL_NAME=colorChannelName
-ALBEDO_CHANNEL_NAME=albedoChannelName
-NORMAL_CHANNEL_NAME=normalChannelName
-BLOOM_PSF_FILE=psfFilePath

Note there mustn't be any space characters!
All file's resolutions must match!
Also you must not put another data like comments or behaviour will be undefined, the app'll crash (TODO: not crash on parse failure)

Description and usage: 

COLOR_FILE: File containing color image data, it's required to run within.
ALBEDO_FILE: File containing albedo image data, it is not required to run within.
NORMAL_FILE: File containing normal image data, it is not required to run within, but it can only be specified if ALBEDO_FILE has been specified.

COLOR_CHANNEL_NAME: Channel name for multilayered EXR images, a layer with most matching name will be chosen (one that contains the channel name as a substring at the earliest position).
ALBEDO_CHANNEL_NAME: Channel name for multilayered EXR images, a layer with most matching name will be chosen (one that contains the channel name as a substring at the earliest position). It can only be specified if ALBEDO_FILE has been specified.
NORMAL_CHANNEL_NAME: Channel name for multilayered EXR images, a layer with most matching name will be chosen (one that contains the channel name as a substring at the earliest position). It can only be specified if NORMAL_FILE has been specified.

For example, given a color image having loaded albedo and normal images as well, you can force color image to use albedo's image data. To perform it, you have to write as following:
COLOR_CHANNEL_NAME=albedo
and it will use the albedo image as color assuming, that there is a valid albedo channel assigned to albedo image - otherwise the default one will be choosen.

CAMERA_TRANSFORM: path to Mitsuba file or direct view matrix values val1,val2,val3,val4,val5,val6,val7,val8,val9

MEDIAN_FILTER_RADIUS: a radius in pixels, valid values are 0 (no filter), 1 and 2. Anything larger than 2 is invalid.

DENOISER_EXPOSURE_BIAS: by telling the OptiX AI denoiser that the image is darker than it actually is you can trick the AI into denoising more aggressively.
It won't brighten your image or anything, we revert the exposure back after the denoise. If you want to influence the final brightness of the image,
you should use the Tonemapping Operator's Key Value.

DENOISER_BLEND_FACTOR: denoiser blend factor, 0.0 is full denoise, 1.0 is no denoise.

BLOOM_FOV: Field of View of the image, its used to scale the size of the Bloom's light-flares (point spread function) to match the Camera's projection.
If you don't want bloom then either provide a PSF image the size of the input image with a single bright pixel, or set the FoV to negative or NaN.
Please note that the Bloom is not implemented yet, so the value is ignored (forced to NaN).

TONEMAPPER: tonemapper - choose between "REINHARD", "ACES" and "NONE". After specifying it you have to pass arguments to revelant tonemapper,
the first argument is always the Key Value, a good default is 0.18 like the Reinhard paper's default.
The "NONE" tonemapper does not take any further arguments and will ignore them, it will simply output the linear light colors to 
a HDR EXR as-is (although auto-exposed), this is useful for further processing in e.g. Photoshop.

The second argument has different meanings for the different tonemappers:
- For "REINHARD" it means "burn out level relative to exposed 1.0 luma level"
basically if after adjusting the key of the image (scaling all pixels so the average luminance value equals the key value) the luma value
is above this burn out parameter, then it is allowed to clip to white, a good default is something like 16.0
- For "ACES" it means "gamma/contrast", a value of 1.0 should leave your contrast as is, but the filmic response curves for tonemapping wash-out the 
colors loosing saturation in our optinion so we recommend to use a value between 0.72 and 0.87 instead.

To wrap up, specifing "REINHARD" tonemapper looks like:
-TONEMAPPER=REINHARD=key,whiteLevel
and specifing "ACES" looks like:
-TONEMAPPER=ACES=key,gamma

OUTPUT: output file with specified extension 
BLOOM_PSF_FILE: A EXR file with a HDR sprite corresponding to the Point Spread Function you want to convolve the image with,
if this file is not provided then we use a built-in PSF as the kernel for the convolution.
)";

constexpr std::string_view COLOR_FILE = "COLOR_FILE";
constexpr std::string_view CAMERA_TRANSFORM = "CAMERA_TRANSFORM";
constexpr std::string_view MEDIAN_FILTER_RADIUS = "MEDIAN_FILTER_RADIUS";
constexpr std::string_view DENOISER_EXPOSURE_BIAS = "DENOISER_EXPOSURE_BIAS";
constexpr std::string_view DENOISER_BLEND_FACTOR = "DENOISER_BLEND_FACTOR";
constexpr std::string_view BLOOM_FOV = "BLOOM_FOV";
constexpr std::string_view TONEMAPPER = "TONEMAPPER";
constexpr std::string_view REINHARD = "REINHARD";
constexpr std::string_view ACES = "ACES";
constexpr std::string_view NONE = "NONE";
constexpr std::string_view OUTPUT = "OUTPUT";

constexpr std::string_view ALBEDO_FILE = "ALBEDO_FILE";
constexpr std::string_view NORMAL_FILE = "NORMAL_FILE";
constexpr std::string_view COLOR_CHANNEL_NAME = "COLOR_CHANNEL_NAME";
constexpr std::string_view ALBEDO_CHANNEL_NAME = "ALBEDO_CHANNEL_NAME";
constexpr std::string_view NORMAL_CHANNEL_NAME = "NORMAL_CHANNEL_NAME";
constexpr std::string_view BLOOM_PSF_FILE = "BLOOM_PSF_FILE";

constexpr std::array<std::string_view, MANDATORY_CMD_ARGUMENTS_AMOUNT> REQUIRED_PARAMETERS =
{
	COLOR_FILE,
	CAMERA_TRANSFORM,
	MEDIAN_FILTER_RADIUS,
	DENOISER_EXPOSURE_BIAS,
	DENOISER_BLEND_FACTOR,
	BLOOM_FOV,
	TONEMAPPER,
	OUTPUT
};

enum DENOISER_TONEMAPPER_EXAMPLE_ARGUMENTS
{
	/*
		Mandatory parameters
	*/

	DTEA_COLOR_FILE,
	DTEA_CAMERA_TRANSFORM,
	DTEA_MEDIAN_FILTER_RADIUS,
	DTEA_DENOISER_EXPOSURE_BIAS,
	DTEA_DENOISER_BLEND_FACTOR,
	DTEA_BLOOM_FOV,
	DTEA_TONEMAPPER,
	DTEA_TONEMAPPER_REINHARD,
	DTEA_TONEMAPPER_ACES,
	DTEA_TONEMAPPER_NONE,
	DTEA_OUTPUT,

	/*
		Optional parameters
	*/

	DTEA_ALBEDO_FILE,
	DTEA_NORMAL_FILE,
	DTEA_COLOR_CHANNEL_NAME,
	DTEA_ALBEDO_CHANNEL_NAME,
	DTEA_NORMAL_CHANNEL_NAME,
	DTEA_BLOOM_PSF_FILE,

	DTEA_COUNT
};

enum TONEMAPPER_ARGUMENTS 
{
	TA_KEY_VALUE,
	TA_EXTRA_PARAMETER,

	TA_COUNT
};


using cmdVariableName = std::string;
using rawValuesOfcmdVariable = std::string;
using variablesType = std::unordered_map<DENOISER_TONEMAPPER_EXAMPLE_ARGUMENTS, std::optional<irr::core::vector<rawValuesOfcmdVariable>>>;

class CommandLineHandler
{
	public:

		CommandLineHandler(irr::core::vector<std::string> argv, irr::asset::IAssetManager* am);

		auto getInputFilesAmount()
		{
			return rawVariables.size();
		}

		auto& getColorFileNameBundle() const
		{
			return colorFileNameBundle;
		}

		auto& getAlbedoFileNameBundle() const
		{
			return albedoFileNameBundle;
		}

		auto& getNormalFileNameBundle() const
		{
			return normalFileNameBundle;
		}

		auto& getColorChannelNameBundle() const
		{
			return colorChannelNameBundle;
		}

		auto& getAlbedoChannelNameBundle() const
		{
			return albedoChannelNameBundle;
		}
		
		auto& getNormalChannelNameBundle() const
		{
			return normalChannelNameBundle;
		}

		auto& getCameraTransformBundle() const
		{
			return cameraTransformBundle;
		}

		auto& getMedianFilterRadiusBundle() const
		{
			return medianFilterRadiusBundle;
		}

		auto& getExposureBiasBundle() const
		{
			return denoiserExposureBiasBundle;
		}

		auto& getDenoiserBlendFactorBundle() const
		{
			return denoiserBlendFactorBundle;
		}

		auto& getBloomFovBundle() const
		{
			return bloomFovBundle;
		}

		auto& getTonemapperBundle() const
		{
			return tonemapperBundle;
		}

		auto& getOutputFileBundle() const
		{
			return outputFileNameBundle;
		}

		auto& getBloomPsfBundle() const
		{
			return bloomPsfFileNameBundle;
		}

		auto getStatus() { return status; }
		auto getMode() { return mode; }
		auto doesItSupportManyInputFiles() { return mode == CLM_BATCH_INPUT; }

	private:

		void initializeMatchingMap(variablesType& rawVariablesPerFile)
		{
			rawVariablesPerFile[DTEA_COLOR_FILE];
			rawVariablesPerFile[DTEA_CAMERA_TRANSFORM];
			rawVariablesPerFile[DTEA_MEDIAN_FILTER_RADIUS];
			rawVariablesPerFile[DTEA_DENOISER_EXPOSURE_BIAS];
			rawVariablesPerFile[DTEA_DENOISER_BLEND_FACTOR];
			rawVariablesPerFile[DTEA_BLOOM_FOV];
			rawVariablesPerFile[DTEA_TONEMAPPER_REINHARD];
			rawVariablesPerFile[DTEA_TONEMAPPER_ACES];
			rawVariablesPerFile[DTEA_TONEMAPPER_NONE];
			rawVariablesPerFile[DTEA_OUTPUT];

			rawVariablesPerFile[DTEA_ALBEDO_FILE];
			rawVariablesPerFile[DTEA_NORMAL_FILE];
			rawVariablesPerFile[DTEA_COLOR_CHANNEL_NAME];
			rawVariablesPerFile[DTEA_ALBEDO_CHANNEL_NAME];
			rawVariablesPerFile[DTEA_NORMAL_CHANNEL_NAME];
			rawVariablesPerFile[DTEA_BLOOM_PSF_FILE];
		}

		DENOISER_TONEMAPPER_EXAMPLE_ARGUMENTS getMatchedVariableMapID(const std::string& variableName)
		{
			if (variableName == COLOR_FILE)
				return DTEA_COLOR_FILE;
			else if (variableName == CAMERA_TRANSFORM)
				return DTEA_CAMERA_TRANSFORM;
			else if (variableName == MEDIAN_FILTER_RADIUS)
				return DTEA_MEDIAN_FILTER_RADIUS;
			else if (variableName == DENOISER_EXPOSURE_BIAS)
				return DTEA_DENOISER_EXPOSURE_BIAS;
			else if (variableName == DENOISER_BLEND_FACTOR)
				return DTEA_DENOISER_BLEND_FACTOR;
			else if (variableName == BLOOM_FOV)
				return DTEA_BLOOM_FOV;
			else if (variableName == TONEMAPPER)
				return DTEA_TONEMAPPER;
			else if (variableName == REINHARD)
				return DTEA_TONEMAPPER_REINHARD;
			else if(variableName == ACES)
				return DTEA_TONEMAPPER_ACES;
			else if(variableName == NONE)
				return DTEA_TONEMAPPER_NONE;
			else if (variableName == OUTPUT)
				return DTEA_OUTPUT;
			else if (variableName == ALBEDO_FILE)
				return DTEA_ALBEDO_FILE;
			else if (variableName == NORMAL_FILE)
				return DTEA_NORMAL_FILE;
			else if (variableName == COLOR_CHANNEL_NAME)
				return DTEA_COLOR_CHANNEL_NAME;
			else if (variableName == ALBEDO_CHANNEL_NAME)
				return DTEA_ALBEDO_CHANNEL_NAME;
			else if (variableName == NORMAL_CHANNEL_NAME)
				return DTEA_NORMAL_CHANNEL_NAME;
			else if (variableName == BLOOM_PSF_FILE)
				return DTEA_BLOOM_PSF_FILE;
			else
				return DTEA_COUNT;
		}

		bool validateMandatoryParameters(const variablesType& rawVariablesPerFile, const size_t idOfInput);

		/*
			Mandatory parameters must have a value. Since they are validated in code,
			there is no need for checking it's content.
		*/

		auto getColorFileName(uint64_t id = 0)
		{
			return rawVariables[id][DTEA_COLOR_FILE].value()[0];
		}
		
		irr::core::matrix3x4SIMD getCameraTransform(uint64_t id = 0);

		auto getMedianFilterRadius(uint64_t id = 0)
		{
			return std::stof(rawVariables[id][DTEA_MEDIAN_FILTER_RADIUS].value()[0]);
		}

		auto getDenoiserExposureBias(uint64_t id = 0)
		{
			return std::stof(rawVariables[id][DTEA_DENOISER_EXPOSURE_BIAS].value()[0]);
		}

		auto getDenoiserBlendFactor(uint64_t id = 0)
		{
			return std::stof(rawVariables[id][DTEA_DENOISER_BLEND_FACTOR].value()[0]);
		}

		auto getBloomFov(uint64_t id = 0)
		{
			return std::stof(rawVariables[id][DTEA_BLOOM_FOV].value()[0]);
		}

		auto getTonemapper(uint64_t id = 0)
		{
			irr::core::vector<float> values;

			uint32_t j = DTEA_TONEMAPPER_REINHARD;
			DENOISER_TONEMAPPER_EXAMPLE_ARGUMENTS num;
			for (; j<=DTEA_TONEMAPPER_NONE; j++)
			{
				num = (DENOISER_TONEMAPPER_EXAMPLE_ARGUMENTS)j;
				if (rawVariables[id][num].has_value())
					break;
			}
			
			if (j<=DTEA_TONEMAPPER_NONE)
			{
				const auto& stringVec = rawVariables[id][num].value();
				for (const auto& str : stringVec)
					values.push_back(std::stof(str));
			}
			
			return std::make_pair(num, values);
		}

		auto getOutputFile(uint64_t id = 0)
		{
			return rawVariables[id][DTEA_OUTPUT].value()[0];
		}

		/*
			Optional parameters don't have to contain any value.
		*/

		std::optional<std::string> getAlbedoFileName(uint64_t id = 0)
		{
			bool ableToReturn = rawVariables[id][DTEA_ALBEDO_FILE].has_value();
			if (ableToReturn)
				return rawVariables[id][DTEA_ALBEDO_FILE].value()[0];
			else
				return {};
		}

		std::optional<std::string> getNormalFileName(uint64_t id = 0);

		std::optional<std::string> getColorChannelName(uint64_t id = 0)
		{
			bool ableToReturn = rawVariables[id][DTEA_COLOR_CHANNEL_NAME].has_value();
			if(ableToReturn)
				return rawVariables[id][DTEA_COLOR_CHANNEL_NAME].value()[0];
			else
				return {};
		}

		std::optional<std::string> getAlbedoChannelName(uint64_t id = 0)
		{
			bool ableToReturn = rawVariables[id][DTEA_ALBEDO_CHANNEL_NAME].has_value() && rawVariables[id][DTEA_ALBEDO_FILE].has_value();
			if (ableToReturn)
				return rawVariables[id][DTEA_ALBEDO_CHANNEL_NAME].value()[0];
			else
				return {};
		}

		std::optional<std::string> getNormalChannelName(uint64_t id = 0)
		{
			bool ableToReturn = rawVariables[id][DTEA_NORMAL_CHANNEL_NAME].has_value() && rawVariables[id][DTEA_NORMAL_FILE].has_value();
			if(ableToReturn)
				return rawVariables[id][DTEA_NORMAL_CHANNEL_NAME].value()[0];
			else
				return {};
		}

		std::optional<std::string> getBloomPsfFile(uint64_t id = 0)
		{
			bool ableToReturn = rawVariables[id][DTEA_BLOOM_PSF_FILE].has_value();
			if (ableToReturn)
				return rawVariables[id][DTEA_BLOOM_PSF_FILE].value()[0];
			else
				return {};
		}

		void performFInalAssignmentStepForUsefulVariables()
		{
			const auto inputFilesAmount = getInputFilesAmount();
			colorFileNameBundle.reserve(inputFilesAmount);
			albedoFileNameBundle.reserve(inputFilesAmount);
			normalFileNameBundle.reserve(inputFilesAmount);
			colorChannelNameBundle.reserve(inputFilesAmount);
			albedoChannelNameBundle.reserve(inputFilesAmount);
			normalChannelNameBundle.reserve(inputFilesAmount);
			cameraTransformBundle.reserve(inputFilesAmount);
			medianFilterRadiusBundle.reserve(inputFilesAmount);
			denoiserExposureBiasBundle.reserve(inputFilesAmount);
			denoiserBlendFactorBundle.reserve(inputFilesAmount);
			bloomFovBundle.reserve(inputFilesAmount);
			tonemapperBundle.reserve(inputFilesAmount);
			outputFileNameBundle.reserve(inputFilesAmount);
			bloomPsfFileNameBundle.reserve(inputFilesAmount);

			for (auto i = 0ul; i < inputFilesAmount; ++i)
			{
				colorFileNameBundle.push_back(getColorFileName(i));
				albedoFileNameBundle.push_back(getAlbedoFileName(i));
				normalFileNameBundle.push_back(getNormalFileName(i));
				colorChannelNameBundle.push_back(getColorChannelName(i));
				albedoChannelNameBundle.push_back(getAlbedoChannelName(i));
				normalChannelNameBundle.push_back(getNormalChannelName(i));
				cameraTransformBundle.push_back(getCameraTransform(i));
				medianFilterRadiusBundle.push_back(getMedianFilterRadius(i));
				denoiserExposureBiasBundle.push_back(getDenoiserExposureBias(i));
				denoiserBlendFactorBundle.push_back(getDenoiserBlendFactor(i));
				bloomFovBundle.push_back(getBloomFov(i));
				tonemapperBundle.push_back(getTonemapper(i));
				outputFileNameBundle.push_back(getOutputFile(i));
				bloomPsfFileNameBundle.push_back(getBloomPsfFile(i));
			}
		}

		bool status;
		COMMAND_LINE_MODE mode;
		irr::core::vector<variablesType> rawVariables;
		irr::asset::IAssetManager * const assetManager;

		// I want to deduce those types bellow by using type from functions above
		// like deduce type of getTonemapper()

		irr::core::vector<std::optional<std::string>> colorFileNameBundle;
		irr::core::vector<std::optional<std::string>> albedoFileNameBundle;
		irr::core::vector<std::optional<std::string>> normalFileNameBundle;
		irr::core::vector<std::optional<std::string>> colorChannelNameBundle;
		irr::core::vector<std::optional<std::string>> albedoChannelNameBundle;
		irr::core::vector<std::optional<std::string>> normalChannelNameBundle;
		irr::core::vector<std::optional<irr::core::matrix3x4SIMD>> cameraTransformBundle;
		irr::core::vector<std::optional<float>> medianFilterRadiusBundle;
		irr::core::vector<std::optional<float>> denoiserExposureBiasBundle;
		irr::core::vector<std::optional<float>> denoiserBlendFactorBundle;
		irr::core::vector<std::optional<float>> bloomFovBundle;
		irr::core::vector<std::pair<DENOISER_TONEMAPPER_EXAMPLE_ARGUMENTS,irr::core::vector<float>>> tonemapperBundle;
		irr::core::vector<std::optional<std::string>> outputFileNameBundle;
		irr::core::vector<std::optional<std::string>> bloomPsfFileNameBundle;

		std::chrono::nanoseconds elapsedTimeXmls = {};
		std::chrono::nanoseconds elapsedTimeEntireLoading = {};
};

#endif // _DENOISER_TONEMAPPER_COMMAND_LINE_HANDLER_