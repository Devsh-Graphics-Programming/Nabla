#ifndef _DENOISER_TONEMAPPER_COMMAND_LINE_HANDLER_
#define _DENOISER_TONEMAPPER_COMMAND_LINE_HANDLER_

#include <iostream>
#include <cstdio>
#include <irrlicht.h>

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
-COLOR_FILE=colorFilename
-CAMERA_TRANSFORM=value,value,value,...
-MEDIAN_FILTER_RADIUS=value
-DENOISER_EXPOSURE_BIAS=value
-DENOISER_BLEND_FACTOR=value
-BLOOM_FOV=theta
-TONEMAPPER=tonemapper=keyValue,extraParameter
-OUTPUT=file.choosenextension
Optional Parameters:
-ALBEDO_FILE=albedoFilename
-NORMAL_FILE=normalFilename
-COLOR_CHANNEL_NAME=colorChannelName
-ALBEDO_CHANNEL_NAME=albedoChannelName
-NORMAL_CHANNEL_NAME=normalChannelName
-BLOOM_PSF_FILE=psfFilename

Note there mustn't be any space characters!
All file's resolutions must match!
Also you must not put another data like comments or behaviour will be undefined, the app'll crash (TODO: not crash on parse failure)

Description and usage: 

COLOR_FILE: File containing color image data, it's required to run within.
ALBEDO_FILE: File containing albedo image data, it is not required to run within.
NORMAL_FILE: File containing normal image data, it is not required to run within, but it can only be specified if ALBEDO_FILE has been specified.

COLOR_CHANNEL_NAME: Channel name of an image for color image that the image will be assigned to. 
ALBEDO_CHANNEL_NAME: Channel name of an image for albedo image that the image will be assigned to. It can only be specified if ALBEDO_FILE has been specified.
NORMAL_CHANNEL_NAME: Channel name of an image for normal image that the image will be assigned to. It can only be specified if NORMAL_FILE has been specified.

For example, given a color image having loaded albedo and normal images as well, you can force color image to use albedo's image data. To perform it, you have to write as following:
COLOR_CHANNEL_NAME=albedo
and it will use the albedo image as color assuming, that there is a valid albedo channel assigned to albedo image - otherwise the default one will be choosen.

CAMERA_TRANSFORM: values as "initializer list" for camera transform matrix with
row_major layout (max 9 values - extra values will be ignored)

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
constexpr std::string_view OUTPUT = "OUTPUT";

constexpr std::string_view ALBEDO_FILE = "ALBEDO_FILE";
constexpr std::string_view NORMAL_FILE = "NORMAL_FILE";
constexpr std::string_view COLOR_CHANNEL_NAME = "COLOR_CHANNEL_NAME";
constexpr std::string_view ALBEDO_CHANNEL_NAME = "ALBEDO_CHANNEL_NAME";
constexpr std::string_view NORMAL_CHANNEL_NAME = "NORMAL_CHANNEL_NAME";
constexpr std::string_view BLOOM_PSF_FILE = "BLOOM_PSF_FILE";

/*
	Those are reserved for files and variables being invalid.
	- INVALID_VARIABLE means that the variable hasn't been specified (optional) or there is something wrong with the variable and so can't be used.
	- INVALID_FILE means that the file doesn't meet the requirements (extension) so can't be used.
*/

constexpr std::string_view INVALID_VARIABLE = "INVALID_VARIABLE";
constexpr std::string_view INVALID_FILE = "INVALID_FILE";

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
	DTEA_REINHARD,
	DTEA_ACES,
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
using variablesType = std::array<std::pair<cmdVariableName, irr::core::vector<rawValuesOfcmdVariable>>, DTEA_COUNT>;

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
			rawVariablesPerFile[DTEA_COLOR_FILE].first = COLOR_FILE;
			rawVariablesPerFile[DTEA_CAMERA_TRANSFORM].first = CAMERA_TRANSFORM;
			rawVariablesPerFile[DTEA_MEDIAN_FILTER_RADIUS].first = MEDIAN_FILTER_RADIUS;
			rawVariablesPerFile[DTEA_DENOISER_EXPOSURE_BIAS].first = DENOISER_EXPOSURE_BIAS;
			rawVariablesPerFile[DTEA_DENOISER_BLEND_FACTOR].first = DENOISER_BLEND_FACTOR;
			rawVariablesPerFile[DTEA_BLOOM_FOV].first = BLOOM_FOV;
			rawVariablesPerFile[DTEA_REINHARD].first = REINHARD;
			rawVariablesPerFile[DTEA_ACES].first = ACES;
			rawVariablesPerFile[DTEA_OUTPUT].first = OUTPUT;

			rawVariablesPerFile[DTEA_ALBEDO_FILE].first = ALBEDO_FILE;
			rawVariablesPerFile[DTEA_NORMAL_FILE].first = NORMAL_FILE;
			rawVariablesPerFile[DTEA_COLOR_CHANNEL_NAME].first = COLOR_CHANNEL_NAME;
			rawVariablesPerFile[DTEA_ALBEDO_CHANNEL_NAME].first = ALBEDO_CHANNEL_NAME;
			rawVariablesPerFile[DTEA_NORMAL_CHANNEL_NAME].first = NORMAL_CHANNEL_NAME;
			rawVariablesPerFile[DTEA_BLOOM_PSF_FILE].first = BLOOM_PSF_FILE;
		}

		auto getColorFileName(uint64_t id = 0)
		{
			return rawVariables[id][DTEA_COLOR_FILE].second[0];
		}

		auto getAlbedoFileName(uint64_t id = 0)
		{
			return rawVariables[id][DTEA_ALBEDO_FILE].second[0];
		}

		auto getNormalFileName(uint64_t id = 0)
		{
			return rawVariables[id][DTEA_NORMAL_FILE].second[0];
		}

		auto getColorChannelName(uint64_t id = 0)
		{
			return rawVariables[id][DTEA_COLOR_CHANNEL_NAME].second[0];
		}

		auto getAlbedoChannelName(uint64_t id = 0)
		{
			return rawVariables[id][DTEA_ALBEDO_CHANNEL_NAME].second[0];
		}

		auto getNormalChannelName(uint64_t id = 0)
		{
			return rawVariables[id][DTEA_NORMAL_CHANNEL_NAME].second[0];
		}

		auto getCameraTransform(uint64_t id = 0)
		{
			irr::core::matrix3x4SIMD cameraTransform;
			const auto send = rawVariables[id][DTEA_CAMERA_TRANSFORM].second.end();
			auto sit = rawVariables[id][DTEA_CAMERA_TRANSFORM].second.begin();
			for (auto i=0; i<3u&&sit!=send; i++)
			for (auto j=0; j<3u&&sit!=send; j++)
				cameraTransform(i,j) = std::stof(*(sit++));

			return cameraTransform;
		}

		auto getMedianFilterRadius(uint64_t id = 0)
		{
			return std::stof(rawVariables[id][DTEA_MEDIAN_FILTER_RADIUS].second[0]);
		}

		auto getDenoiserExposureBias(uint64_t id = 0)
		{
			return std::stof(rawVariables[id][DTEA_DENOISER_EXPOSURE_BIAS].second[0]);
		}

		auto getDenoiserBlendFactor(uint64_t id = 0)
		{
			return std::stof(rawVariables[id][DTEA_DENOISER_BLEND_FACTOR].second[0]);
		}

		auto getBloomFov(uint64_t id = 0)
		{
			return std::stof(rawVariables[id][DTEA_BLOOM_FOV].second[0]);
		}

		auto getTonemapper(uint64_t id = 0)
		{
			const bool isChoosenReinhard = rawVariables[id][DTEA_ACES].second.empty();
			auto tonemapper = isChoosenReinhard ? rawVariables[id][DTEA_REINHARD] : rawVariables[id][DTEA_ACES];
			irr::core::vector<float> values(TA_COUNT);

			for (auto i = 0; i < TA_COUNT; ++i)
				*(values.begin() + i) = std::stof(tonemapper.second[i]);

			return std::make_pair(tonemapper.first, values);
		}

		auto getOutputFile(uint64_t id = 0)
		{
			return rawVariables[id][DTEA_OUTPUT].second[0];
		}

		auto getBloomPsfFile(uint64_t id = 0)
		{
			return rawVariables[id][DTEA_BLOOM_PSF_FILE].second[0];
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

		// I want to deduce those types bellow by using type from functions above
		// like deduce type of getTonemapper()

		irr::core::vector<std::string> colorFileNameBundle;
		irr::core::vector<std::string> albedoFileNameBundle;
		irr::core::vector<std::string> normalFileNameBundle;
		irr::core::vector<std::string> colorChannelNameBundle;
		irr::core::vector<std::string> albedoChannelNameBundle;
		irr::core::vector<std::string> normalChannelNameBundle;
		irr::core::vector<irr::core::matrix3x4SIMD> cameraTransformBundle;
		irr::core::vector<float> medianFilterRadiusBundle;
		irr::core::vector<float> denoiserExposureBiasBundle;
		irr::core::vector<float> denoiserBlendFactorBundle;
		irr::core::vector<float> bloomFovBundle;
		irr::core::vector<std::pair<std::string, irr::core::vector<float>>> tonemapperBundle;
		irr::core::vector<std::string> outputFileNameBundle;
		irr::core::vector<std::string> bloomPsfFileNameBundle;
};

#endif // _DENOISER_TONEMAPPER_COMMAND_LINE_HANDLER_