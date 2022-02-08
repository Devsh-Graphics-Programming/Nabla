// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _DENOISER_TONEMAPPER_COMMAND_LINE_HANDLER_
#define _DENOISER_TONEMAPPER_COMMAND_LINE_HANDLER_

#include <iostream>
#include <cstdio>
#include <chrono>
#include "nabla.h"
#include "nbl/core/core.h"
#include "nbl/ext/MitsubaLoader/CMitsubaLoader.h"

#define PROPER_CMD_ARGUMENTS_AMOUNT 17
#define MANDATORY_CMD_ARGUMENTS_AMOUNT 9
#define OPTIONAL_CMD_ARGUMENTS_AMOUNT 5
#define PROPER_BATCH_FILE_ARGUMENTS_AMOUNT 3

enum COMMAND_LINE_MODE
{
	CLM_UNKNOWN,
	CLM_CMD_LIST,
	CLM_BATCH_INPUT,

	CLM_COUNT
};

constexpr std::string_view requiredArgumentsMessage = R"(
Pass appropriate arguments to launch the example or load them using predefined file!
* To load them with a arguments file usage type DenoiserTonemapper.exe -batch <yourargumentfile.txt>
* To load them passing arguments through cmd.

Mandatory parameters:
-COLOR_FILE=colorFilePath
-CAMERA_TRANSFORM=mitsubaFilePath or val1,val2,val3,...,val9
-DENOISER_EXPOSURE_BIAS=value
-DENOISER_BLEND_FACTOR=value
-BLOOM_PSF_FILE=psfFilePath
-BLOOM_RELATIVE_SCALE=value
-BLOOM_INTENSITY=value
-TONEMAPPER=tonemapper=keyValue,extraParameter
-OUTPUT=file.choosenextension
Optional Parameters:
-ALBEDO_FILE=albedoFilePath
-NORMAL_FILE=normalFilePath
-COLOR_CHANNEL_NAME=colorChannelName
-ALBEDO_CHANNEL_NAME=albedoChannelName
-NORMAL_CHANNEL_NAME=normalChannelName

Note there mustn't be any space characters!
All files' (except the bloom kernel) resolutions must match!
Also you must not put another data like comments or behaviour will be undefined, the app'll crash (TODO: not crash on parse failure)

Description and usage: 

COLOR_FILE: File containing color image data, it's required to run with.
ALBEDO_FILE: File containing albedo image data, it is not required to run.
NORMAL_FILE: File containing normal image data, it is not required to run, but it can only be specified if ALBEDO_FILE has been specified.

COLOR_CHANNEL_NAME: Channel name for multilayered EXR images, a layer with most matching name will be chosen (one that contains the channel name as a substring at the earliest position).
ALBEDO_CHANNEL_NAME: Channel name for multilayered EXR images, a layer with most matching name will be chosen (one that contains the channel name as a substring at the earliest position). It can only be specified if ALBEDO_FILE has been specified.
NORMAL_CHANNEL_NAME: Channel name for multilayered EXR images, a layer with most matching name will be chosen (one that contains the channel name as a substring at the earliest position). It can only be specified if NORMAL_FILE has been specified.

For example, given a color image having loaded albedo and normal images as well, you can force color image to use albedo's image data. To perform it, you have to write as following:
COLOR_CHANNEL_NAME=albedo
and it will use the albedo image as color assuming, that there is a valid albedo channel assigned to albedo image - otherwise the default one will be choosen.

CAMERA_TRANSFORM: path to Mitsuba file or direct view matrix values val1,val2,val3,val4,val5,val6,val7,val8,val9

DENOISER_EXPOSURE_BIAS: by telling the OptiX AI denoiser that the image is darker than it actually is you can trick the AI into denoising more aggressively.
It won't brighten your image or anything, we revert the exposure back after the denoise. If you want to influence the final brightness of the image,
you should use the Tonemapping Operator's Key Value.

DENOISER_BLEND_FACTOR: denoiser blend factor, 0.0 is full denoise, 1.0 is no denoise.

BLOOM_PSF_FILE: A EXR file with a HDR sprite corresponding to the Point Spread Function you want to convolve the image with.

BLOOM_RELATIVE_SCALE: Must not be negative or greater than 1. The scale relative to the kernel being placed at the center of the denoised image and isotropically stretched until it touches one of the sides of the denoised image.
You'll usually want to keep this value quite small (below 1/32) to make sure the kernel has a higher pixel density relative to the image, otherwise you'll end up blurring the image (the executable will print a warning).
Do not use it for actually controlling the size of the flare, this is because even though the kernel is scaled, it is normalized to conserve energy, which eventually degenerates into an ugly box blur as scale tends to 0.

BLOOM_INTENSITY: Must be in the [0,1] range, it actually controls the size of the flare much better than the relative scale (assuming a good HDR kernel with very long tails).
If you don't want bloom then set bloom intensity to 0.

TONEMAPPER: tonemapper - choose between "REINHARD", "ACES" and "NONE". After specifying it you have to pass arguments to revelant tonemapper,
the first argument is always the Key Value, a good default is 0.18 like the Reinhard paper's default.
The "NONE" tonemapper does not take any further arguments and will ignore them, it will simply output the linear light colors to 
a HDR EXR as-is (although auto-exposed), this is useful for further processing in e.g. Photoshop.
The NONE tonemapper can also have a special key value of "AutoexposureOff", which makes sure the exposure of the render will not be altered after denoising.

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
while "NONE looks like:
-TONEMAPPER=NONE=key
or:
-TONEMAPPER=NONE=AutoexposureOff

OUTPUT: output file with specified extension 
The kernel must be centered and in RGB or RGBA floating point format. Resolution should be less than the denoised image.
If this file is not provided then we use a built-in PSF as the kernel for the convolution.
)";

constexpr std::string_view COLOR_FILE = "COLOR_FILE";
constexpr std::string_view CAMERA_TRANSFORM = "CAMERA_TRANSFORM";
constexpr std::string_view DENOISER_EXPOSURE_BIAS = "DENOISER_EXPOSURE_BIAS";
constexpr std::string_view DENOISER_BLEND_FACTOR = "DENOISER_BLEND_FACTOR";
constexpr std::string_view BLOOM_PSF_FILE = "BLOOM_PSF_FILE";
constexpr std::string_view BLOOM_RELATIVE_SCALE = "BLOOM_RELATIVE_SCALE";
constexpr std::string_view BLOOM_INTENSITY = "BLOOM_INTENSITY";
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

constexpr std::array<std::string_view, MANDATORY_CMD_ARGUMENTS_AMOUNT> REQUIRED_PARAMETERS =
{
	COLOR_FILE,
	CAMERA_TRANSFORM,
	DENOISER_EXPOSURE_BIAS,
	DENOISER_BLEND_FACTOR,
	BLOOM_PSF_FILE,
	BLOOM_RELATIVE_SCALE,
	BLOOM_INTENSITY,
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
	DTEA_DENOISER_EXPOSURE_BIAS,
	DTEA_DENOISER_BLEND_FACTOR,
	DTEA_BLOOM_PSF_FILE,
	DTEA_BLOOM_RELATIVE_SCALE,
	DTEA_BLOOM_INTENSITY,
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
using variablesType = std::unordered_map<DENOISER_TONEMAPPER_EXAMPLE_ARGUMENTS, std::optional<nbl::core::vector<rawValuesOfcmdVariable>>>;

class CommandLineHandler
{
	public:

		CommandLineHandler(nbl::core::vector<std::string> argv, nbl::asset::IAssetManager* am, nbl::io::IFileSystem* fs);

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

		auto& getExposureBiasBundle() const
		{
			return denoiserExposureBiasBundle;
		}

		auto& getDenoiserBlendFactorBundle() const
		{
			return denoiserBlendFactorBundle;
		}

		auto& getBloomRelativeScaleBundle() const
		{
			return bloomRelativeScaleBundle;
		}

		auto& getBloomIntensityBundle() const
		{
			return bloomIntensityBundle;
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
			rawVariablesPerFile[DTEA_DENOISER_EXPOSURE_BIAS];
			rawVariablesPerFile[DTEA_DENOISER_BLEND_FACTOR];
			rawVariablesPerFile[DTEA_BLOOM_PSF_FILE];
			rawVariablesPerFile[DTEA_BLOOM_RELATIVE_SCALE];
			rawVariablesPerFile[DTEA_BLOOM_INTENSITY];
			rawVariablesPerFile[DTEA_TONEMAPPER_REINHARD];
			rawVariablesPerFile[DTEA_TONEMAPPER_ACES];
			rawVariablesPerFile[DTEA_TONEMAPPER_NONE];
			rawVariablesPerFile[DTEA_OUTPUT];

			rawVariablesPerFile[DTEA_ALBEDO_FILE];
			rawVariablesPerFile[DTEA_NORMAL_FILE];
			rawVariablesPerFile[DTEA_COLOR_CHANNEL_NAME];
			rawVariablesPerFile[DTEA_ALBEDO_CHANNEL_NAME];
			rawVariablesPerFile[DTEA_NORMAL_CHANNEL_NAME];
		}

		DENOISER_TONEMAPPER_EXAMPLE_ARGUMENTS getMatchedVariableMapID(const std::string& variableName)
		{
			if (variableName == COLOR_FILE)
				return DTEA_COLOR_FILE;
			else if (variableName == CAMERA_TRANSFORM)
				return DTEA_CAMERA_TRANSFORM;
			else if (variableName == DENOISER_EXPOSURE_BIAS)
				return DTEA_DENOISER_EXPOSURE_BIAS;
			else if (variableName == DENOISER_BLEND_FACTOR)
				return DTEA_DENOISER_BLEND_FACTOR;
			else if (variableName == BLOOM_PSF_FILE)
				return DTEA_BLOOM_PSF_FILE;
			else if (variableName == BLOOM_RELATIVE_SCALE)
				return DTEA_BLOOM_RELATIVE_SCALE;
			else if (variableName == BLOOM_INTENSITY)
				return DTEA_BLOOM_INTENSITY;
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
		
		nbl::core::matrix3x4SIMD getCameraTransform(uint64_t id = 0);

		auto getDenoiserExposureBias(uint64_t id = 0)
		{
			return std::stof(rawVariables[id][DTEA_DENOISER_EXPOSURE_BIAS].value()[0]);
		}

		auto getDenoiserBlendFactor(uint64_t id = 0)
		{
			return std::stof(rawVariables[id][DTEA_DENOISER_BLEND_FACTOR].value()[0]);
		}

		auto getBloomRelativeScale(uint64_t id = 0)
		{
			return std::stof(rawVariables[id][DTEA_BLOOM_RELATIVE_SCALE].value()[0]);
		}

		auto getBloomIntensity(uint64_t id = 0)
		{
			return std::stof(rawVariables[id][DTEA_BLOOM_INTENSITY].value()[0]);
		}

		auto getTonemapper(uint64_t id = 0)
		{
			nbl::core::vector<float> values;

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
					values.push_back(str=="AutoexposureOff" ? nbl::core::nan<float>():std::stof(str));
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
			denoiserExposureBiasBundle.reserve(inputFilesAmount);
			denoiserBlendFactorBundle.reserve(inputFilesAmount);
			bloomPsfFileNameBundle.reserve(inputFilesAmount);
			bloomRelativeScaleBundle.reserve(inputFilesAmount);
			bloomIntensityBundle.reserve(inputFilesAmount);
			tonemapperBundle.reserve(inputFilesAmount);
			outputFileNameBundle.reserve(inputFilesAmount);

			for (auto i = 0ul; i < inputFilesAmount; ++i)
			{
				colorFileNameBundle.push_back(getColorFileName(i));
				albedoFileNameBundle.push_back(getAlbedoFileName(i));
				normalFileNameBundle.push_back(getNormalFileName(i));
				colorChannelNameBundle.push_back(getColorChannelName(i));
				albedoChannelNameBundle.push_back(getAlbedoChannelName(i));
				normalChannelNameBundle.push_back(getNormalChannelName(i));
				cameraTransformBundle.push_back(getCameraTransform(i));
				denoiserExposureBiasBundle.push_back(getDenoiserExposureBias(i));
				denoiserBlendFactorBundle.push_back(getDenoiserBlendFactor(i));
				bloomPsfFileNameBundle.push_back(getBloomPsfFile(i));
				bloomRelativeScaleBundle.push_back(getBloomRelativeScale(i));
				bloomIntensityBundle.push_back(getBloomIntensity(i));
				tonemapperBundle.push_back(getTonemapper(i));
				outputFileNameBundle.push_back(getOutputFile(i));
			}
		}

		bool status;
		COMMAND_LINE_MODE mode;
		nbl::core::vector<variablesType> rawVariables;
		nbl::asset::IAssetManager * const assetManager;

		// I want to deduce those types bellow by using type from functions above
		// like deduce type of getTonemapper()

		nbl::core::vector<std::optional<std::string>> colorFileNameBundle;
		nbl::core::vector<std::optional<std::string>> albedoFileNameBundle;
		nbl::core::vector<std::optional<std::string>> normalFileNameBundle;
		nbl::core::vector<std::optional<std::string>> colorChannelNameBundle;
		nbl::core::vector<std::optional<std::string>> albedoChannelNameBundle;
		nbl::core::vector<std::optional<std::string>> normalChannelNameBundle;
		nbl::core::vector<std::optional<nbl::core::matrix3x4SIMD>> cameraTransformBundle;
		nbl::core::vector<std::optional<float>> denoiserExposureBiasBundle;
		nbl::core::vector<std::optional<float>> denoiserBlendFactorBundle;
		nbl::core::vector<std::optional<std::string>> bloomPsfFileNameBundle;
		nbl::core::vector<std::optional<float>> bloomRelativeScaleBundle;
		nbl::core::vector<std::optional<float>> bloomIntensityBundle;
		nbl::core::vector<std::pair<DENOISER_TONEMAPPER_EXAMPLE_ARGUMENTS,nbl::core::vector<float>>> tonemapperBundle;
		nbl::core::vector<std::optional<std::string>> outputFileNameBundle;

		std::chrono::nanoseconds elapsedTimeXmls = {};
		std::chrono::nanoseconds elapsedTimeEntireLoading = {};
};

#endif // _DENOISER_TONEMAPPER_COMMAND_LINE_HANDLER_