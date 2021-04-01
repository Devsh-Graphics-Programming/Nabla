// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>

#include "CommandLineHandler.hpp"
#include "nbl/asset/filters/dithering/CPrecomputedDither.h"

#include "nbl/ext/ToneMapper/CToneMapper.h"
#include "nbl/ext/FFT/FFT.h"
#include "nbl/ext/OptiX/Manager.h"

#include "CommonPushConstants.h"

using namespace nbl;
using namespace asset;
using namespace video;

enum E_IMAGE_INPUT : uint32_t
{
	EII_COLOR,
	EII_ALBEDO,
	EII_NORMAL,
	EII_COUNT
};
struct ImageToDenoise
{
	uint32_t width = 0u, height = 0u;
	uint32_t colorTexelSize = 0u;
	E_IMAGE_INPUT denoiserType = EII_COUNT;
	float bloomScale;
	core::smart_refctd_ptr<asset::ICPUImage> image[EII_COUNT] = { nullptr,nullptr,nullptr };
	core::smart_refctd_ptr<asset::ICPUImage> kernel = nullptr;
};
struct DenoiserToUse
{
	core::smart_refctd_ptr<ext::OptiX::IDenoiser> m_denoiser;
	size_t stateOffset = 0u;
	size_t stateSize = 0u;
	size_t scratchSize = 0u;
};

int error_code = 0;
bool check_error(bool cond, const char* message)
{
	error_code++;
	if (cond)
		os::Printer::log(message, ELL_ERROR);
	return cond;
}

constexpr uint32_t overlap = 64;
//constexpr uint32_t tileWidth = 1920/2, tileHeight = 1080/2;
constexpr uint32_t tileWidth = 1024, tileHeight = 1024;
constexpr uint32_t outputDimensions[] = { tileWidth ,tileHeight };
constexpr uint32_t outputDimensionsWithOverlap[] = { tileWidth+overlap*2,tileHeight+overlap*2 };

int main(int argc, char* argv[])
{
	nbl::SIrrlichtCreationParameters params;
	params.Bits = 24;
	params.ZBufferBits = 24;
	params.DriverType = video::EDT_OPENGL;
	params.WindowSize = core::dimension2d<uint32_t>(1280, 720);
	params.Fullscreen = false;
	params.Vsync = true;
	params.Doublebuffer = true;
	params.Stencilbuffer = false;
	params.StreamingDownloadBufferSize = 1024*1024*1024; // for 16k images
	auto device = createDeviceEx(params);

	if (check_error(!device,"Could not create Irrlicht Device!"))
		return error_code;

	auto driver = device->getVideoDriver();
	auto smgr = device->getSceneManager();
	auto am = device->getAssetManager();

	auto compiler = am->getGLSLCompiler();
	auto filesystem = device->getFileSystem();

	auto getArgvFetchedList = [&]()
	{
		core::vector<std::string> arguments;
		arguments.reserve(PROPER_CMD_ARGUMENTS_AMOUNT);
		arguments.emplace_back(argv[0]);
		if (argc>1)
			for (auto i = 1ul; i < argc; ++i)
				arguments.emplace_back(argv[i]);
		else
		{
			arguments.emplace_back("-batch");
			arguments.emplace_back("../exampleInputArguments.txt");
		}

		return arguments;
	};
	
	auto cmdHandler = CommandLineHandler(getArgvFetchedList(), am, device->getFileSystem());

	if (check_error(!cmdHandler.getStatus(),"Could not parse input commands!"))
		return error_code;

	auto m_optixManager = ext::OptiX::Manager::create(driver,device->getFileSystem());
	if (check_error(!m_optixManager, "Could not initialize CUDA or OptiX!"))
		return error_code;
	auto m_cudaStream = m_optixManager->getDeviceStream(0);
	if (check_error(!m_cudaStream, "Could not obtain CUDA stream!"))
		return error_code;
	auto m_optixContext = m_optixManager->createContext(0);
	if (check_error(!m_optixContext, "Could not create Optix Context!"))
		return error_code;

	constexpr auto forcedOptiXFormat = OPTIX_PIXEL_FORMAT_HALF3; // TODO: make more denoisers with formats
	E_FORMAT nblFmtRequired = EF_UNKNOWN;
	switch (forcedOptiXFormat)
	{
		case OPTIX_PIXEL_FORMAT_UCHAR3:
			nblFmtRequired = EF_R8G8B8_SRGB;
			break;
		case OPTIX_PIXEL_FORMAT_UCHAR4:
			nblFmtRequired = EF_R8G8B8A8_SRGB;
			break;
		case OPTIX_PIXEL_FORMAT_HALF3:
			nblFmtRequired = EF_R16G16B16_SFLOAT;
			break;
		case OPTIX_PIXEL_FORMAT_HALF4:
			nblFmtRequired = EF_R16G16B16A16_SFLOAT;
			break;
		case OPTIX_PIXEL_FORMAT_FLOAT3:
			nblFmtRequired = EF_R32G32B32_SFLOAT;
			break;
		case OPTIX_PIXEL_FORMAT_FLOAT4:
			nblFmtRequired = EF_R32G32B32A32_SFLOAT;
			break;
	}
	constexpr auto forcedOptiXFormatPixelStride = 6u;
	DenoiserToUse denoisers[EII_COUNT];
	{
		OptixDenoiserOptions opts = { OPTIX_DENOISER_INPUT_RGB };
		denoisers[EII_COLOR].m_denoiser = m_optixContext->createDenoiser(&opts);
		if (check_error(!denoisers[EII_COLOR].m_denoiser, "Could not create Optix Color Denoiser!"))
			return error_code;
		opts.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO;
		denoisers[EII_ALBEDO].m_denoiser = m_optixContext->createDenoiser(&opts);
		if (check_error(!denoisers[EII_ALBEDO].m_denoiser, "Could not create Optix Color-Albedo Denoiser!"))
			return error_code;
		opts.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL;
		denoisers[EII_NORMAL].m_denoiser = m_optixContext->createDenoiser(&opts);
		if (check_error(!denoisers[EII_NORMAL].m_denoiser, "Could not create Optix Color-Albedo-Normal Denoiser!"))
			return error_code;
	}


	using FFTClass = ext::FFT::FFT;
	using LumaMeterClass = ext::LumaMeter::CLumaMeter;
	using ToneMapperClass = ext::ToneMapper::CToneMapper;

	constexpr uint32_t kComputeWGSize = FFTClass::DEFAULT_WORK_GROUP_SIZE; // if it changes, maybe it breaks stuff
	
	constexpr bool usingHalfFloatFFTStorage = false;
	constexpr bool usingLumaMeter = true;
	constexpr auto MeterMode = LumaMeterClass::EMM_MEDIAN;
	const auto HistogramBufferSize = LumaMeterClass::getOutputBufferSize(MeterMode);
	constexpr float lowerPercentile = 0.45f;
	constexpr float upperPercentile = 0.55f;
	constexpr auto TMO = ToneMapperClass::EO_ACES;
	auto histogramBuffer = driver->createDeviceLocalGPUBufferOnDedMem(HistogramBufferSize);
	// clear the histogram to 0s
	driver->fillBuffer(histogramBuffer.get(),0u,HistogramBufferSize,0u);

	constexpr auto SharedDescriptorSetDescCount = 4u;
	core::smart_refctd_ptr<IGPUDescriptorSetLayout> sharedDescriptorSetLayout;
	core::smart_refctd_ptr<IGPUPipelineLayout> sharedPipelineLayout;
	core::smart_refctd_ptr<IGPUComputePipeline> deinterleavePipeline,intensityPipeline,secondLumaMeterAndFirstFFTPipeline,interleavePipeline;
	{
		auto deinterleaveShader = driver->createGPUShader(core::make_smart_refctd_ptr<ICPUShader>(R"===(
#version 450 core
#extension GL_EXT_shader_16bit_storage : require
#define _NBL_GLSL_EXT_LUMA_METER_FIRST_PASS_DEFINED_
#include "../ShaderCommon.glsl"
layout(binding = 0, std430) restrict readonly buffer ImageInputBuffer
{
	f16vec4 inBuffer[];
};
layout(binding = 1, std430) restrict writeonly buffer ImageOutputBuffer
{
	float16_t outBuffer[];
};
vec3 fetchData(in uvec3 texCoord)
{
	vec3 data = vec4(inBuffer[pc.data.inImageTexelOffset[texCoord.z]+texCoord.y*pc.data.inImageTexelPitch[texCoord.z]+texCoord.x]).xyz;
	bool invalid = any(isnan(data))||any(isinf(abs(data)));
	if (texCoord.z==EII_ALBEDO)
		data = invalid ? vec3(1.0):data;
	else if (texCoord.z==EII_NORMAL)
	{
		data = invalid||length(data)<0.000000001 ? vec3(0.0,0.0,1.0):normalize(pc.data.normalMatrix*data);
	}
	return data;
}
void main()
{
	globalPixelData = fetchData(gl_GlobalInvocationID);
	bool colorLayer = gl_GlobalInvocationID.z==EII_COLOR;
	if (colorLayer)
	{
		nbl_glsl_ext_LumaMeter(colorLayer && gl_GlobalInvocationID.x<pc.data.imageWidth);
		barrier(); // no barrier because we were just reading from shared not writing since the last memory barrier
	}
	repackBuffer[gl_LocalInvocationIndex*SHARED_CHANNELS+0u] = floatBitsToUint(globalPixelData[0u]);
	repackBuffer[gl_LocalInvocationIndex*SHARED_CHANNELS+1u] = floatBitsToUint(globalPixelData[1u]);
	repackBuffer[gl_LocalInvocationIndex*SHARED_CHANNELS+2u] = floatBitsToUint(globalPixelData[2u]);
	barrier();
	memoryBarrierShared();
	const uint outImagePitch = pc.data.imageWidth*SHARED_CHANNELS;
	uint rowOffset = pc.data.outImageOffset[gl_GlobalInvocationID.z]+gl_GlobalInvocationID.y*outImagePitch;
	uint lineOffset = gl_WorkGroupID.x*COMPUTE_WG_SIZE*SHARED_CHANNELS+gl_LocalInvocationIndex;
	if (lineOffset<outImagePitch)
		outBuffer[rowOffset+lineOffset] = float16_t(uintBitsToFloat(repackBuffer[gl_LocalInvocationIndex+COMPUTE_WG_SIZE*0u]));
	lineOffset += COMPUTE_WG_SIZE;
	if (lineOffset<outImagePitch)
		outBuffer[rowOffset+lineOffset] = float16_t(uintBitsToFloat(repackBuffer[gl_LocalInvocationIndex+COMPUTE_WG_SIZE*1u]));
	lineOffset += COMPUTE_WG_SIZE;
	if (lineOffset<outImagePitch)
		outBuffer[rowOffset+lineOffset] = float16_t(uintBitsToFloat(repackBuffer[gl_LocalInvocationIndex+COMPUTE_WG_SIZE*2u]));
}
		)==="));
		auto intensityShader = driver->createGPUShader(core::make_smart_refctd_ptr<ICPUShader>(R"===(
#version 450 core
#extension GL_EXT_shader_16bit_storage : require
#include "../ShaderCommon.glsl"
layout(set=_NBL_GLSL_EXT_LUMA_METER_OUTPUT_SET_DEFINED_, binding=_NBL_GLSL_EXT_LUMA_METER_OUTPUT_BINDING_DEFINED_) restrict readonly buffer LumaMeterOutputBuffer
{
	nbl_glsl_ext_LumaMeter_output_t lumaParams[];
};
layout(binding = 3, std430) restrict writeonly buffer IntensityBuffer
{
	float intensity[];
};

int nbl_glsl_ext_LumaMeter_getCurrentLumaOutputOffset()
{
	return pc.data.beforeDenoise!=0u ? 0:1;
}
nbl_glsl_ext_LumaMeter_output_SPIRV_CROSS_is_dumb_t nbl_glsl_ext_ToneMapper_getLumaMeterOutput()
{
	nbl_glsl_ext_LumaMeter_output_SPIRV_CROSS_is_dumb_t retval;
	retval = lumaParams[nbl_glsl_ext_LumaMeter_getCurrentLumaOutputOffset()].packedHistogram[gl_LocalInvocationIndex];
	for (int i=1; i<_NBL_GLSL_EXT_LUMA_METER_BIN_GLOBAL_REPLICATION; i++)
		retval += lumaParams[nbl_glsl_ext_LumaMeter_getCurrentLumaOutputOffset()].packedHistogram[gl_LocalInvocationIndex+i*_NBL_GLSL_EXT_LUMA_METER_BIN_COUNT];
	return retval;
}
void main()
{
	const bool firstInvocation = all(equal(uvec3(0,0,0),gl_GlobalInvocationID));
	const bool beforeDenoise = pc.data.beforeDenoise!=0u;
	const bool autoexposureOn = pc.data.autoexposureOff==0u;

	float optixIntensity = 1.0;
	if (beforeDenoise||autoexposureOn)
	{
		nbl_glsl_ext_LumaMeter_PassInfo_t lumaPassInfo;
		lumaPassInfo.percentileRange[0] = pc.data.percentileRange[0];
		lumaPassInfo.percentileRange[1] = pc.data.percentileRange[1];
		float measuredLumaLog2 = nbl_glsl_ext_LumaMeter_getMeasuredLumaLog2(nbl_glsl_ext_ToneMapper_getLumaMeterOutput(),lumaPassInfo);
		if (firstInvocation)
		{
			measuredLumaLog2 += beforeDenoise ? pc.data.denoiserExposureBias:0.0;
			optixIntensity = nbl_glsl_ext_LumaMeter_getOptiXIntensity(measuredLumaLog2);
		}
	}
	
	if (firstInvocation)
		intensity[pc.data.intensityBufferDWORDOffset] = optixIntensity;
}
		)==="));
		auto secondLumaMeterAndFirstFFTShader = driver->createGPUShader(core::make_smart_refctd_ptr<ICPUShader>(R"===(
#version 450 core
#extension GL_EXT_shader_16bit_storage : require
#define _NBL_GLSL_EXT_LUMA_METER_FIRST_PASS_DEFINED_
#include "../ShaderCommon.glsl"
layout(binding = 0, std430) restrict readonly buffer ImageInputBuffer
{
	float16_t inBuffer[];
};
void main()
{
	uint dataOffset = pc.data.outImageOffset[EII_COLOR]+(gl_GlobalInvocationID.y*pc.data.imageWidth+gl_GlobalInvocationID.x)*SHARED_CHANNELS;

	// TODO: Optimize this fetch
	globalPixelData = vec3(inBuffer[dataOffset+0u],inBuffer[dataOffset+1u],inBuffer[dataOffset+2u]);

	nbl_glsl_ext_LumaMeter(gl_GlobalInvocationID.x<pc.data.imageWidth);
	barrier();
}
		)==="));
		auto interleaveShader = driver->createGPUShader(core::make_smart_refctd_ptr<ICPUShader>(R"===(
#version 450 core
#extension GL_EXT_shader_16bit_storage : require
#include "../ShaderCommon.glsl"
#include "nbl/builtin/glsl/ext/ToneMapper/operators.glsl"
layout(binding = 0, std430) restrict readonly buffer ImageInputBuffer
{
	float16_t inBuffer[];
};
layout(binding = 1, std430) restrict writeonly buffer ImageOutputBuffer
{
	f16vec4 outBuffer[];
};
layout(binding = 3, std430) restrict readonly buffer IntensityBuffer
{
	float intensity[];
};
void main()
{
	uint wgOffset = pc.data.outImageOffset[EII_COLOR]+(gl_GlobalInvocationID.y*pc.data.imageWidth+gl_WorkGroupID.x*COMPUTE_WG_SIZE)*SHARED_CHANNELS;
	uint localOffset = gl_LocalInvocationIndex;
	repackBuffer[localOffset] = floatBitsToUint(float(inBuffer[wgOffset+localOffset]));
	localOffset += COMPUTE_WG_SIZE;
	repackBuffer[localOffset] = floatBitsToUint(float(inBuffer[wgOffset+localOffset]));
	localOffset += COMPUTE_WG_SIZE;
	repackBuffer[localOffset] = floatBitsToUint(float(inBuffer[wgOffset+localOffset]));
	barrier();
	memoryBarrierShared();
	bool alive = gl_GlobalInvocationID.x<pc.data.imageWidth;
	vec3 color = uintBitsToFloat(uvec3(repackBuffer[gl_LocalInvocationIndex*SHARED_CHANNELS+0u],repackBuffer[gl_LocalInvocationIndex*SHARED_CHANNELS+1u],repackBuffer[gl_LocalInvocationIndex*SHARED_CHANNELS+2u]));
	
	color = _NBL_GLSL_EXT_LUMA_METER_XYZ_CONVERSION_MATRIX_DEFINED_*color;
	color *= intensity[pc.data.intensityBufferDWORDOffset]; // *= 0.18/AvgLuma
	switch (pc.data.tonemappingOperator)
	{
		case _NBL_GLSL_EXT_TONE_MAPPER_REINHARD_OPERATOR:
		{
			nbl_glsl_ext_ToneMapper_ReinhardParams_t tonemapParams;
			tonemapParams.keyAndManualLinearExposure = pc.data.tonemapperParams[0];
			tonemapParams.rcpWhite2 = pc.data.tonemapperParams[1];
			color = nbl_glsl_ext_ToneMapper_Reinhard(tonemapParams,color);
			break;
		}
		case _NBL_GLSL_EXT_TONE_MAPPER_ACES_OPERATOR:
		{
			nbl_glsl_ext_ToneMapper_ACESParams_t tonemapParams;
			tonemapParams.gamma = pc.data.tonemapperParams[0];
			tonemapParams.exposure = pc.data.tonemapperParams[1];
			color = nbl_glsl_ext_ToneMapper_ACES(tonemapParams,color);
			break;
		}
		default:
		{
			color *= pc.data.tonemapperParams[0];
			break;
		}
	}
	color = nbl_glsl_XYZtosRGB*color;
	// TODO: compute DFFT of the image in the X-axis
	uint dataOffset = pc.data.inImageTexelOffset[EII_COLOR]+gl_GlobalInvocationID.y*pc.data.inImageTexelPitch[EII_COLOR]+gl_GlobalInvocationID.x;
	if (alive)
		outBuffer[dataOffset] = f16vec4(vec4(color,1.0));
}
		)==="));
		struct SpecializationConstants
		{
			uint32_t workgroupSize = kComputeWGSize;
			uint32_t enumEII_COLOR = EII_COLOR;
			uint32_t enumEII_ALBEDO = EII_ALBEDO;
			uint32_t enumEII_NORMAL = EII_NORMAL;
		} specData;
		auto specConstantBuffer = core::make_smart_refctd_ptr<CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t> > >(sizeof(SpecializationConstants), &specData, core::adopt_memory);
		IGPUSpecializedShader::SInfo specInfo = {	core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IGPUSpecializedShader::SInfo::SMapEntry> >
													(
														std::initializer_list<IGPUSpecializedShader::SInfo::SMapEntry>
														{
															{0u,offsetof(SpecializationConstants,workgroupSize),sizeof(SpecializationConstants::workgroupSize)},
															{1u,offsetof(SpecializationConstants,enumEII_COLOR),sizeof(SpecializationConstants::enumEII_COLOR)},
															{2u,offsetof(SpecializationConstants,enumEII_ALBEDO),sizeof(SpecializationConstants::enumEII_ALBEDO)},
															{3u,offsetof(SpecializationConstants,enumEII_NORMAL),sizeof(SpecializationConstants::enumEII_NORMAL)}
														}
													),
													core::smart_refctd_ptr(specConstantBuffer),"main",ISpecializedShader::ESS_COMPUTE
												};
		auto deinterleaveSpecializedShader = driver->createGPUSpecializedShader(deinterleaveShader.get(),specInfo);
		auto intensitySpecializedShader = driver->createGPUSpecializedShader(intensityShader.get(),specInfo);
		auto secondLumaMeterAndFirstFFTSpecializedShader = driver->createGPUSpecializedShader(secondLumaMeterAndFirstFFTShader.get(),specInfo);
		auto interleaveSpecializedShader = driver->createGPUSpecializedShader(interleaveShader.get(),specInfo);
		
		IGPUDescriptorSetLayout::SBinding binding[SharedDescriptorSetDescCount] = {
			{0u,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
			{1u,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
			{2u,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
			{3u,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr}
		};
		sharedDescriptorSetLayout = driver->createGPUDescriptorSetLayout(binding,binding+SharedDescriptorSetDescCount);
		SPushConstantRange pcRange[1] = {IGPUSpecializedShader::ESS_COMPUTE,0u,sizeof(CommonPushConstants)};
		sharedPipelineLayout = driver->createGPUPipelineLayout(pcRange,pcRange+sizeof(pcRange)/sizeof(SPushConstantRange),core::smart_refctd_ptr(sharedDescriptorSetLayout));

		deinterleavePipeline = driver->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(sharedPipelineLayout),std::move(deinterleaveSpecializedShader));
		intensityPipeline = driver->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(sharedPipelineLayout),std::move(intensitySpecializedShader));
		secondLumaMeterAndFirstFFTPipeline = driver->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(sharedPipelineLayout),std::move(secondLumaMeterAndFirstFFTSpecializedShader));
		interleavePipeline = driver->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(sharedPipelineLayout),std::move(interleaveSpecializedShader));
	}

	const auto inputFilesAmount = cmdHandler.getInputFilesAmount();
	const auto& colorFileNameBundle = cmdHandler.getColorFileNameBundle();
	const auto& albedoFileNameBundle = cmdHandler.getAlbedoFileNameBundle();
	const auto& normalFileNameBundle = cmdHandler.getNormalFileNameBundle();
	const auto& colorChannelNameBundle = cmdHandler.getColorChannelNameBundle();
	const auto& albedoChannelNameBundle = cmdHandler.getAlbedoChannelNameBundle();
	const auto& normalChannelNameBundle = cmdHandler.getNormalChannelNameBundle();
	const auto& cameraTransformBundle = cmdHandler.getCameraTransformBundle();
	const auto& denoiserExposureBiasBundle = cmdHandler.getExposureBiasBundle();
	const auto& denoiserBlendFactorBundle = cmdHandler.getDenoiserBlendFactorBundle();
	const auto& bloomScaleBundle = cmdHandler.getBloomScaleBundle();
	const auto& tonemapperBundle = cmdHandler.getTonemapperBundle();
	const auto& outputFileBundle = cmdHandler.getOutputFileBundle();
	const auto& bloomPsfFileBundle = cmdHandler.getBloomPsfBundle();

	auto makeImageIDString = [](uint32_t i, const core::vector<std::optional<std::string>>& fileNameBundle = {})
	{
		std::string imageIDString("Image Input #");
		imageIDString += std::to_string(i);

		if (!fileNameBundle.empty() && fileNameBundle[i].has_value())
		{
			imageIDString += " called \"";
			imageIDString += fileNameBundle[i].value();
		}

		return imageIDString;
	};

	core::vector<ImageToDenoise> images(inputFilesAmount);
	// load images
	uint32_t maxResolution[2] = { 0,0 };
	{
		asset::IAssetLoader::SAssetLoadParams lp(0ull,nullptr);
		auto default_kernel_image_bundle = am->getAsset("../../media/kernels/physical_flare_512.exr",lp); // TODO: use builtins?

		for (size_t i=0; i < inputFilesAmount; i++)
		{
			const auto imageIDString = makeImageIDString(i, colorFileNameBundle);

			auto color_image_bundle = am->getAsset(colorFileNameBundle[i].value(), lp); decltype(color_image_bundle) albedo_image_bundle, normal_image_bundle;
			if (color_image_bundle.getContents().empty())
			{
				os::Printer::log("ERROR (" + std::to_string(__LINE__) + " line): Could not load the image from file: " + imageIDString + "!", ELL_ERROR);
				continue;
			}

			albedo_image_bundle = albedoFileNameBundle[i].has_value() ? am->getAsset(albedoFileNameBundle[i].value(), lp) : decltype(albedo_image_bundle)();
			normal_image_bundle = normalFileNameBundle[i].has_value() ? am->getAsset(normalFileNameBundle[i].value(), lp) : decltype(normal_image_bundle)();

			auto kernel_image_bundle = bloomPsfFileBundle[i].has_value() ? am->getAsset(bloomPsfFileBundle[i].value(),lp):default_kernel_image_bundle;

			auto& outParam = images[i];

			auto getImageAssetGivenChannelName = [](asset::SAssetBundle& assetBundle, const std::optional<std::string>& channelName) -> core::smart_refctd_ptr<ICPUImage>
			{
				if (assetBundle.getContents().empty())
					return nullptr;

				// calculate a score for how much each channel name matches the requested
				size_t firstChannelNameOccurence = std::string::npos;
				uint32_t pickedChannel = 0u;
				auto contents = assetBundle.getContents();
				if (channelName.has_value())
					for (auto& asset : contents)
					{
						assert(asset);
						
						const auto* bundleMeta = assetBundle.getMetadata();
						const auto* exrmeta = static_cast<const COpenEXRMetadata*>(bundleMeta);
						const auto* metadata = static_cast<const COpenEXRMetadata::CImage*>(exrmeta->getAssetSpecificMetadata(core::smart_refctd_ptr_static_cast<ICPUImage>(asset).get()));

						if (strcmp(exrmeta->getLoaderName(), COpenEXRMetadata::LoaderName) != 0)
							continue;
						else
						{
							const auto& assetMetaChannelName = metadata->m_name;
							auto found = assetMetaChannelName.find(channelName.value());
							if (found >= firstChannelNameOccurence)
								continue;
							firstChannelNameOccurence = found;
							pickedChannel = std::distance(contents.begin(), &asset);
						}
					}

				return asset::IAsset::castDown<ICPUImage>(contents.begin()[pickedChannel]);
			};

			auto color = getImageAssetGivenChannelName(color_image_bundle,colorChannelNameBundle[i]);
			decltype(color) albedo = getImageAssetGivenChannelName(albedo_image_bundle,albedoChannelNameBundle[i]);
			decltype(color) normal = getImageAssetGivenChannelName(normal_image_bundle,normalChannelNameBundle[i]);

			decltype(color) kernel = getImageAssetGivenChannelName(kernel_image_bundle,{});
			if (!kernel)
			{
				kernel = getImageAssetGivenChannelName(default_kernel_image_bundle,{});
				if (!kernel)
				{
					os::Printer::log(imageIDString+"Could not load default Bloom Kernel Image, denoise will be skipped!", ELL_ERROR);
					continue;
				}
			}

			auto putImageIntoImageToDenoise = [&](asset::SAssetBundle& queriedBundle, core::smart_refctd_ptr<ICPUImage>&& queriedImage, E_IMAGE_INPUT defaultEII, const std::optional<std::string>& actualWantedChannel)
			{
				outParam.image[defaultEII] = nullptr;
				if (!queriedImage)
				{
					switch (defaultEII)
					{
						case EII_ALBEDO:
						{
							os::Printer::log("INFO (" + std::to_string(__LINE__) + " line): Running in mode without albedo channel!", ELL_INFORMATION);
						} break;
						case EII_NORMAL:
						{
							os::Printer::log("INFO (" + std::to_string(__LINE__) + " line): Running in mode without normal channel!", ELL_INFORMATION);
						} break;
					}
					return;
				}

				const auto* bundleMeta = queriedBundle.getMetadata();
				const auto* exrmeta = static_cast<const COpenEXRMetadata*>(bundleMeta);
				const auto* metadata = static_cast<const COpenEXRMetadata::CImage*>(exrmeta->getAssetSpecificMetadata(queriedImage.get()));

				if (strcmp(exrmeta->getLoaderName(), COpenEXRMetadata::LoaderName)!=0)
					os::Printer::log("WARNING (" + std::to_string(__LINE__) + "): "+ imageIDString+" is not an EXR file, so there are no multiple layers of channels.", ELL_WARNING);
				else if (!actualWantedChannel.has_value())
					os::Printer::log("WARNING (" + std::to_string(__LINE__) + "): User did not specify channel choice for "+ imageIDString+" using the default (first).", ELL_WARNING);
				else if (metadata->m_name!=actualWantedChannel.value())
				{
					os::Printer::log("WARNING (" + std::to_string(__LINE__) + "): Using best fit channel \""+ metadata->m_name +"\" for requested \""+actualWantedChannel.value()+"\" out of "+ imageIDString+"!", ELL_WARNING);
				}
				outParam.image[defaultEII] = std::move(queriedImage);
			};

			putImageIntoImageToDenoise(color_image_bundle, std::move(color), EII_COLOR, colorChannelNameBundle[i]);
			putImageIntoImageToDenoise(albedo_image_bundle, std::move(albedo), EII_ALBEDO, albedoChannelNameBundle[i]);
			putImageIntoImageToDenoise(normal_image_bundle, std::move(normal), EII_NORMAL, normalChannelNameBundle[i]);
			outParam.kernel = std::move(kernel);
		}
		// check inputs and set-up
		for (size_t i=0; i<inputFilesAmount; i++)
		{
			auto imageIDString = makeImageIDString(i);

			auto& outParam = images[i];
			{
				auto* colorImage = outParam.image[EII_COLOR].get();
				if (!colorImage)
				{
					os::Printer::log(imageIDString+"Could not find the Color Channel for denoising, image will be skipped!", ELL_ERROR);
					outParam = {};
					continue;
				}

				const auto& colorCreationParams = colorImage->getCreationParameters();
				const auto& extent = colorCreationParams.extent;
				// compute storage size and check if we can successfully upload
				{
					auto regions = colorImage->getRegions();
					assert(regions.begin()+1u==regions.end());

					const auto& region = regions.begin()[0];
					assert(region.bufferRowLength);
					outParam.colorTexelSize = asset::getTexelOrBlockBytesize(colorCreationParams.format);
				}
				
				const auto& kerDim = outParam.kernel->getCreationParameters().extent;
				const float bloomScale = core::min(float(extent.width)/float(kerDim.width),float(extent.height)/float(kerDim.height))*bloomScaleBundle[i].value();
				if (bloomScale>1.f)
					os::Printer::log(imageIDString + "Bloom Kernel will Clip and loose sharpness, increase resolution of bloom kernel!", ELL_WARNING);
				const auto marginSrcDim = [extent,kerDim,bloomScale]() -> auto
				{
					auto tmp = extent;
					for (auto i=0u; i<3u; i++)
					{
						const auto coord = (&kerDim.width)[i];
						if (coord>1u)
							(&tmp.width)[i] += core::max(coord*bloomScale,1u)-1u;
					}
					return tmp;
				}();
				constexpr uint32_t colorChannelsFFT = 3u;
				FFTClass::getOutputBufferSize(usingHalfFloatFFTStorage,marginSrcDim,colorChannelsFFT); // TODO: get a max of this value
				{
					// TODO: store these
					FFTClass::Parameters_t fftPushConstants[3];
					FFTClass::DispatchInfo_t fftDispatchInfo[3];
					const ISampler::E_TEXTURE_CLAMP fftPadding[2] = {ISampler::ETC_MIRROR,ISampler::ETC_MIRROR};
					const auto passes = FFTClass::buildParameters(false,colorChannelsFFT,extent,fftPushConstants,fftDispatchInfo,fftPadding,marginSrcDim);
					{
						// override for less work and storage (dont need to store the extra padding of the last axis after iFFT)
						fftPushConstants[1].output_strides.x = fftPushConstants[0].input_strides.x;
						fftPushConstants[1].output_strides.y = fftPushConstants[0].input_strides.y;
						fftPushConstants[1].output_strides.z = fftPushConstants[1].input_strides.z;
						fftPushConstants[1].output_strides.w = fftPushConstants[1].input_strides.w;
						// iFFT
						fftPushConstants[2].input_dimensions = fftPushConstants[1].input_dimensions;
						{
							fftPushConstants[2].input_dimensions.w = fftPushConstants[0].input_dimensions.w^0x80000000u;
							fftPushConstants[2].input_strides = fftPushConstants[1].output_strides;
							fftPushConstants[2].output_strides = fftPushConstants[0].input_strides;
						}
						fftDispatchInfo[2] = fftDispatchInfo[0];
					}
					assert(passes==2);
				}

				outParam.denoiserType = EII_COLOR;

				outParam.width = extent.width;
				outParam.height = extent.height;

				maxResolution[0] = core::max(maxResolution[0], outParam.width);
				maxResolution[1] = core::max(maxResolution[1], outParam.height);
			}

			auto& albedoImage = outParam.image[EII_ALBEDO];
			if (albedoImage)
			{
				auto extent = albedoImage->getCreationParameters().extent;
				if (extent.width!=outParam.width || extent.height!=outParam.height)
				{
					os::Printer::log(imageIDString + "Image extent of the Albedo Channel does not match the Color Channel, Albedo Channel will not be used!", ELL_ERROR);
					albedoImage = nullptr;
				}
				else
					outParam.denoiserType = EII_ALBEDO;
			}

			auto& normalImage = outParam.image[EII_NORMAL];
			if (normalImage)
			{
				auto extent = normalImage->getCreationParameters().extent;
				if (extent.width != outParam.width || extent.height != outParam.height)
				{
					os::Printer::log(imageIDString + "Image extent of the Normal Channel does not match the Color Channel, Normal Channel will not be used!", ELL_ERROR);
					normalImage = nullptr;
				}
				else if (!albedoImage)
				{
					os::Printer::log(imageIDString + "Invalid Albedo Channel for denoising, Normal Channel will not be used!", ELL_ERROR);
					normalImage = nullptr;
				}
				else
					outParam.denoiserType = EII_NORMAL;
			}
		}
	}

#define DENOISER_BUFFER_COUNT 3u //had to change to a define cause lambda was complaining about it not being a constant expression when capturing
	// keep all CUDA links in an array (less code to map/unmap
	cuda::CCUDAHandler::GraphicsAPIObjLink<video::IGPUBuffer> bufferLinks[DENOISER_BUFFER_COUNT];
	// except for the scratch CUDA buffer which can and will be ENORMOUS
	CUdeviceptr denoiserScratch = 0ull;
	// set-up denoisers
	constexpr size_t IntensityValuesSize = sizeof(float);
	auto& intensityBuffer = bufferLinks[0];
	auto& denoiserState = bufferLinks[0];
	auto& temporaryPixelBuffer = bufferLinks[1];
	auto& imagePixelBuffer = bufferLinks[2]; // buffer to store result of denoising of a tile/image
	size_t denoiserStateBufferSize = 0ull;
	{
		size_t scratchBufferSize = 0ull;
		size_t pixelBufferSize = 0ull;
		for (uint32_t i=0u; i<EII_COUNT; i++)
		{
			auto& denoiser = denoisers[i].m_denoiser;

			OptixDenoiserSizes m_denoiserMemReqs;
			if (denoiser->computeMemoryResources(&m_denoiserMemReqs, outputDimensions)!=OPTIX_SUCCESS)
			{
				static const char* errorMsgs[EII_COUNT] = {	"Failed to compute Color-Denoiser Memory Requirements!",
															"Failed to compute Color-Albedo-Denoiser Memory Requirements!",
															"Failed to compute Color-Albedo-Normal-Denoiser Memory Requirements!"};
				os::Printer::log(errorMsgs[i],ELL_ERROR);
				denoiser = nullptr;
				continue;
			}

			denoisers[i].stateOffset = denoiserStateBufferSize;
			denoiserStateBufferSize += denoisers[i].stateSize = m_denoiserMemReqs.stateSizeInBytes;
			scratchBufferSize = core::max(scratchBufferSize, denoisers[i].scratchSize = m_denoiserMemReqs.withOverlapScratchSizeInBytes);
			pixelBufferSize = core::max(pixelBufferSize, core::max(asset::getTexelOrBlockBytesize(EF_R32G32B32A32_SFLOAT), (i + 1u) * forcedOptiXFormatPixelStride) * maxResolution[0] * maxResolution[1]);

		}
		std::string message = "Total VRAM consumption for Denoiser algorithm: ";
		os::Printer::log(message+std::to_string(denoiserStateBufferSize+scratchBufferSize+pixelBufferSize), ELL_INFORMATION);

		if (check_error(pixelBufferSize==0ull,"No input files at all!"))
			return error_code;

		denoiserState = driver->createDeviceLocalGPUBufferOnDedMem(denoiserStateBufferSize+IntensityValuesSize);
		if (check_error(!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::registerBuffer(&denoiserState)),"Could not register buffer for Denoiser states!"))
			return error_code;
		temporaryPixelBuffer = driver->createDeviceLocalGPUBufferOnDedMem(pixelBufferSize);
		if (check_error(!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::registerBuffer(&temporaryPixelBuffer)),"Could not register buffer for Denoiser scratch memory!"))
			return error_code;
		if (check_error(!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::cuda.pcuMemAlloc_v2(&denoiserScratch,scratchBufferSize)), "Could not register buffer for Denoiser temporary memory with CUDA natively!"))
			return error_code;
	}
	const auto intensityBufferOffset = denoiserStateBufferSize;

	video::CAssetPreservingGPUObjectFromAssetConverter assetConverter(am,driver);
	// do the processing
	for (size_t i=0; i<inputFilesAmount; i++)
	{
		auto& param = images[i];
		if (param.denoiserType>=EII_COUNT)
			continue;
		const auto denoiserInputCount = param.denoiserType+1u;

		// set up the constants (partially)
		CommonPushConstants shaderConstants;
		{
			for (uint32_t j=0u; j<denoiserInputCount; j++)
				shaderConstants.outImageOffset[j] = j*param.width*param.height*forcedOptiXFormatPixelStride/sizeof(uint16_t); // float 16 actually
			shaderConstants.imageWidth = param.width;
			assert(intensityBufferOffset%IntensityValuesSize==0u);
			shaderConstants.beforeDenoise = 1u;

			shaderConstants.intensityBufferDWORDOffset = intensityBufferOffset/IntensityValuesSize;
			shaderConstants.denoiserExposureBias = denoiserExposureBiasBundle[i].value();

			shaderConstants.autoexposureOff = 0u;
			switch (tonemapperBundle[i].first)
			{
				case DTEA_TONEMAPPER_REINHARD:
					shaderConstants.tonemappingOperator = ToneMapperClass::EO_REINHARD;
					break;
				case DTEA_TONEMAPPER_ACES:
					shaderConstants.tonemappingOperator = ToneMapperClass::EO_ACES;
					break;
				case DTEA_TONEMAPPER_NONE:
					shaderConstants.tonemappingOperator = ToneMapperClass::EO_COUNT;
					break;
				default:
					assert(false, "An unexcepted error ocured while trying to specify tonemapper!");
					break;
			}

			float key = tonemapperBundle[i].second[TA_KEY_VALUE];
			const float optiXIntensityKeyCompensation = -log2(0.18);
			float extraParam = tonemapperBundle[i].second[TA_EXTRA_PARAMETER];
			switch (shaderConstants.tonemappingOperator)
			{
				case ToneMapperClass::EO_REINHARD:
				{
					auto tp = ToneMapperClass::Params_t<ToneMapperClass::EO_REINHARD>(optiXIntensityKeyCompensation, key, extraParam);
					shaderConstants.tonemapperParams[0] = tp.keyAndLinearExposure;
					shaderConstants.tonemapperParams[1] = tp.rcpWhite2;
					break;
				}
				case ToneMapperClass::EO_ACES:
				{
					auto tp = ToneMapperClass::Params_t<ToneMapperClass::EO_ACES>(optiXIntensityKeyCompensation, key, extraParam);
					shaderConstants.tonemapperParams[0] = tp.gamma;
					shaderConstants.tonemapperParams[1] = (&tp.gamma)[1];
					break;
				}
				default:
				{
					if (core::isnan(key))
					{
						shaderConstants.tonemapperParams[0] = 0.18;
						shaderConstants.autoexposureOff = 1u;
					}
					else
						shaderConstants.tonemapperParams[0] = key;
					shaderConstants.tonemapperParams[0] *= exp2(optiXIntensityKeyCompensation);
					shaderConstants.tonemapperParams[1] = core::nan<float>();
					break;
				}
			}
			auto totalSampleCount = param.width * param.height;
			shaderConstants.percentileRange[0] = lowerPercentile * float(totalSampleCount);
			shaderConstants.percentileRange[1] = upperPercentile * float(totalSampleCount);
			shaderConstants.normalMatrix = cameraTransformBundle[i].value();
		}

		// upload image channels and register their buffer
		{
			asset::ICPUBuffer* buffersToUpload[EII_COUNT];
			for (uint32_t j=0u; j<denoiserInputCount; j++)
				buffersToUpload[j] = param.image[j]->getBuffer();
			auto gpubuffers = driver->getGPUObjectsFromAssets(buffersToUpload,buffersToUpload+denoiserInputCount,&assetConverter);

			imagePixelBuffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuffers->operator[](0)->getBuffer());
			if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::registerBuffer(&imagePixelBuffer)))
			{
				os::Printer::log(makeImageIDString(i) + "Could register the image data buffer with CUDA, skipping image!", ELL_ERROR);
				continue;
			}

			for (uint32_t j=0u; j<denoiserInputCount; j++)
			{
				auto offsetPair = gpubuffers->operator[](j);
				// it needs to be packed into the same buffer to work
				assert(offsetPair->getBuffer()==imagePixelBuffer.getObject());

				// make sure cache doesn't retain the GPU object paired to CPU object (could have used a custom IGPUObjectFromAssetConverter derived class with overrides to achieve this)
				am->removeCachedGPUObject(buffersToUpload[j],offsetPair);

				auto image = param.image[j];
				const auto& creationParameters = image->getCreationParameters();
				assert(asset::getTexelOrBlockBytesize(creationParameters.format)==param.colorTexelSize);
				// set up some image pitch and offset info
				shaderConstants.inImageTexelPitch[j] = image->getRegions().begin()[0].bufferRowLength;
				shaderConstants.inImageTexelOffset[j] = offsetPair->getOffset();
				assert(shaderConstants.inImageTexelOffset[j]%param.colorTexelSize==0u);
				shaderConstants.inImageTexelOffset[j] /= param.colorTexelSize;
			}
			// upload the constants to the GPU
			driver->pushConstants(sharedPipelineLayout.get(), video::IGPUSpecializedShader::ESS_COMPUTE, 0u, sizeof(CommonPushConstants), &shaderConstants);
		}

		// process
		uint32_t workgroupCounts[2] = {(param.width+kComputeWGSize-1u)/kComputeWGSize,param.height};
		{
			// bind shader resources
			{
				// create descriptor set
				auto descriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr(sharedDescriptorSetLayout));
				// write descriptor set
				{
					IGPUDescriptorSet::SDescriptorInfo infos[SharedDescriptorSetDescCount];
					infos[0].desc = core::smart_refctd_ptr<IGPUBuffer>(imagePixelBuffer.getObject());
					infos[0].buffer = {0u,imagePixelBuffer.getObject()->getMemoryReqs().vulkanReqs.size};
					infos[1].desc = core::smart_refctd_ptr<IGPUBuffer>(temporaryPixelBuffer.getObject());
					infos[1].buffer = {0u,temporaryPixelBuffer.getObject()->getMemoryReqs().vulkanReqs.size};
					infos[2].desc = histogramBuffer;
					infos[2].buffer = {0u,HistogramBufferSize};
					infos[3].desc = core::smart_refctd_ptr<IGPUBuffer>(intensityBuffer.getObject());
					infos[3].buffer = {0u,intensityBuffer.getObject()->getMemoryReqs().vulkanReqs.size};
					IGPUDescriptorSet::SWriteDescriptorSet writes[SharedDescriptorSetDescCount];
					for (uint32_t i=0u; i<SharedDescriptorSetDescCount; i++)
						writes[i] = {descriptorSet.get(),i,0u,1u,EDT_STORAGE_BUFFER,infos+i};
					driver->updateDescriptorSets(SharedDescriptorSetDescCount,writes,0u,nullptr);
				}
				// bind descriptor set (for all shaders)
				driver->bindDescriptorSets(video::EPBP_COMPUTE,sharedPipelineLayout.get(),0u,1u,&descriptorSet.get(),nullptr);
			}
			// compute shader pre-preprocess (transform normals and compute luminosity)
			{
				// bind deinterleave pipeline
				driver->bindComputePipeline(deinterleavePipeline.get());
				// dispatch
				driver->dispatch(workgroupCounts[0],workgroupCounts[1],denoiserInputCount);
				COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
				// bind intensity pipeline
				driver->bindComputePipeline(intensityPipeline.get());
				// dispatch
				driver->dispatch(1u,1u,1u);
				// issue a full memory barrier (or at least all buffer read/write barrier)
				COpenGLExtensionHandler::extGlMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT | GL_PIXEL_BUFFER_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
			}

			// optix processing
			{
				// map buffer
				if (check_error(!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::acquireAndGetPointers(bufferLinks,bufferLinks+DENOISER_BUFFER_COUNT,m_cudaStream)),"Error when mapping OpenGL Buffers to CUdeviceptr!"))
					return error_code;

				auto unmapBuffers = [&m_cudaStream,&bufferLinks]() -> void
				{
					void* scratch[DENOISER_BUFFER_COUNT*sizeof(CUgraphicsResource)];
					if (check_error(!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::releaseResourcesToGraphics(scratch,bufferLinks,bufferLinks+DENOISER_BUFFER_COUNT,m_cudaStream)), "Error when unmapping CUdeviceptr back to OpenGL!"))
						exit(error_code);
				};
#undef DENOISER_BUFFER_COUNT
				core::SRAIIBasedExiter<decltype(unmapBuffers)> exitRoutine(unmapBuffers);

				cuda::CCUDAHandler::GraphicsAPIObjLink<video::IGPUBuffer> fakeScratchLink;
				fakeScratchLink.asBuffer.pointer = denoiserScratch;

				// set up denoiser
				auto& denoiser = denoisers[param.denoiserType];
				if (denoiser.m_denoiser->setup(m_cudaStream, outputDimensionsWithOverlap, denoiserState, denoiser.stateSize, fakeScratchLink, denoiser.scratchSize, denoiser.stateOffset) != OPTIX_SUCCESS)
				{
					os::Printer::log(makeImageIDString(i) + "Could not setup the denoiser for the image resolution and denoiser buffers, skipping image!", ELL_ERROR);
					continue;
				}
				
				//invocation params
				OptixDenoiserParams denoiserParams = {};
				denoiserParams.blendFactor = denoiserBlendFactorBundle[i].value();
				denoiserParams.denoiseAlpha = 0u;
				denoiserParams.hdrIntensity = intensityBuffer.asBuffer.pointer + intensityBufferOffset;

				//input with RGB, Albedo, Normals
				OptixImage2D denoiserInputs[EII_COUNT];
				OptixImage2D denoiserOutput;
				
				for (size_t k = 0; k < denoiserInputCount; k++)
				{
					denoiserInputs[k].data = temporaryPixelBuffer.asBuffer.pointer + shaderConstants.outImageOffset[k] * sizeof(uint16_t);
					denoiserInputs[k].width = param.width;
					denoiserInputs[k].height = param.height;
					denoiserInputs[k].rowStrideInBytes = param.width * forcedOptiXFormatPixelStride;
					denoiserInputs[k].format = forcedOptiXFormat;
					denoiserInputs[k].pixelStrideInBytes = forcedOptiXFormatPixelStride;

				}

				denoiserOutput.data = imagePixelBuffer.asBuffer.pointer + shaderConstants.inImageTexelOffset[EII_COLOR];
				denoiserOutput.width = param.width;
				denoiserOutput.height = param.height;
				denoiserOutput.rowStrideInBytes = param.width * forcedOptiXFormatPixelStride;
				denoiserOutput.format = forcedOptiXFormat;
				denoiserOutput.pixelStrideInBytes = forcedOptiXFormatPixelStride;

				//invoke
				if (denoiser.m_denoiser->tileAndInvoke(
					m_cudaStream,
					&denoiserParams,
					denoiserInputs,
					denoiserInputCount,
					&denoiserOutput,
					fakeScratchLink,
					denoiser.scratchSize,
					overlap,
					tileWidth,
					tileHeight
				) != OPTIX_SUCCESS)
				{
					os::Printer::log(makeImageIDString(i) + "Could not invoke the denoiser sucessfully, skipping image!", ELL_ERROR);
					continue;
				}
			}

			// compute post-processing
			{
				// let the shaders know we're in the second phase now
				shaderConstants.beforeDenoise = 0u;
				driver->pushConstants(sharedPipelineLayout.get(), video::IGPUSpecializedShader::ESS_COMPUTE, offsetof(CommonPushConstants,beforeDenoise), sizeof(uint32_t), &shaderConstants.beforeDenoise);
				// Bloom
				{
					core::smart_refctd_ptr<IGPUImageView> kerImageView;
					{
						auto kerGpuImages = driver->getGPUObjectsFromAssets(&param.kernel,&param.kernel+1u,&assetConverter);


						IGPUImageView::SCreationParams kerImgViewInfo;
						kerImgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
						kerImgViewInfo.image = kerGpuImages->operator[](0u);

						// make sure cache doesn't retain the GPU object paired to CPU object (could have used a custom IGPUObjectFromAssetConverter derived class with overrides to achieve this)
						am->removeCachedGPUObject(param.kernel.get(),kerImgViewInfo.image);

						kerImgViewInfo.viewType = IGPUImageView::ET_2D;
						kerImgViewInfo.format = kerImgViewInfo.image->getCreationParameters().format;
						kerImgViewInfo.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0u);
						kerImgViewInfo.subresourceRange.baseMipLevel = 0;
						kerImgViewInfo.subresourceRange.levelCount = kerImgViewInfo.image->getCreationParameters().mipLevels;
						kerImgViewInfo.subresourceRange.baseArrayLayer = 0;
						kerImgViewInfo.subresourceRange.layerCount = 1;
						kerImageView = driver->createGPUImageView(std::move(kerImgViewInfo));
					}

					driver->bindComputePipeline(secondLumaMeterAndFirstFFTPipeline.get());
					//FFTClass::dispatchHelper(driver, imageFirstFFTPipelineLayout.get(), fftPushConstants[0], fftDispatchInfo[0]);
					// dispatch
					driver->dispatch(workgroupCounts[0],workgroupCounts[1],1u);
					COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

					// TODO: do X-axis pass of the DFFT

					// TODO: multiply the spectra together 

					// TODO: perform inverse Y-axis DFFT and interleave the results

					// bind intensity pipeline
					driver->bindComputePipeline(intensityPipeline.get());
					// dispatch
					driver->dispatch(1u,1u,1u);
					COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
				}
				// Tonemap and interleave the output
				{
					driver->bindComputePipeline(interleavePipeline.get());
					driver->dispatch(workgroupCounts[0],workgroupCounts[1],1u);
					// issue a full memory barrier (or at least all buffer read/write barrier)
					COpenGLExtensionHandler::extGlMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
				}
			}
			// delete descriptor sets (implicit from destructor)
		}

		{
			auto downloadStagingArea = driver->getDefaultDownStreamingBuffer();
			uint32_t address = std::remove_pointer<decltype(downloadStagingArea)>::type::invalid_address; // remember without initializing the address to be allocated to invalid_address you won't get an allocation!

			// image view
			core::smart_refctd_ptr<ICPUImageView> imageView;
			const uint32_t colorBufferBytesize = param.height * param.width * param.colorTexelSize;
			{
				// create image
				ICPUImage::SCreationParams imgParams;
				imgParams.flags = static_cast<ICPUImage::E_CREATE_FLAGS>(0u); // no flags
				imgParams.type = ICPUImage::ET_2D;
				imgParams.format = param.image[EII_COLOR]->getCreationParameters().format;
				imgParams.extent = {param.width,param.height,1u};
				imgParams.mipLevels = 1u;
				imgParams.arrayLayers = 1u;
				imgParams.samples = ICPUImage::ESCF_1_BIT;

				auto image = ICPUImage::create(std::move(imgParams));

				// get the data from the GPU
				{
					constexpr uint64_t timeoutInNanoSeconds = 300000000000u;
					const auto waitPoint = std::chrono::high_resolution_clock::now()+std::chrono::nanoseconds(timeoutInNanoSeconds);

					// download buffer
					{
						const uint32_t alignment = 4096u; // common page size
						auto unallocatedSize = downloadStagingArea->multi_alloc(waitPoint, 1u, &address, &colorBufferBytesize, &alignment);
					
						if (unallocatedSize)
						{
							os::Printer::log(makeImageIDString(i)+"Could not download the buffer from the GPU!",ELL_ERROR);
							continue;
						}

						driver->copyBuffer(temporaryPixelBuffer.getObject(),downloadStagingArea->getBuffer(),0u,address,colorBufferBytesize);
					}
					auto downloadFence = driver->placeFence(true);

					// set up regions
					auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IImage::SBufferCopy> >(1u);
					{
						auto& region = regions->front();
						region.bufferOffset = 0u;
						region.bufferRowLength = param.image[EII_COLOR]->getRegions().begin()[0].bufferRowLength;
						region.bufferImageHeight = param.height;
						//region.imageSubresource.aspectMask = wait for Vulkan;
						region.imageSubresource.mipLevel = 0u;
						region.imageSubresource.baseArrayLayer = 0u;
						region.imageSubresource.layerCount = 1u;
						region.imageOffset = { 0u,0u,0u };
						region.imageExtent = imgParams.extent;
					}
					// the cpu is not touching the data yet because the custom CPUBuffer is adopting the memory (no copy)
					auto* data = reinterpret_cast<uint8_t*>(downloadStagingArea->getBufferPointer())+address;
					auto cpubufferalias = core::make_smart_refctd_ptr<asset::CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t> > >(colorBufferBytesize, data, core::adopt_memory);
					image->setBufferAndRegions(std::move(cpubufferalias),regions);

					// wait for download fence and then invalidate the CPU cache
					{
						auto result = downloadFence->waitCPU(timeoutInNanoSeconds,true);
						if (result==E_DRIVER_FENCE_RETVAL::EDFR_TIMEOUT_EXPIRED||result==E_DRIVER_FENCE_RETVAL::EDFR_FAIL)
						{
							os::Printer::log(makeImageIDString(i)+"Could not download the buffer from the GPU, fence not signalled!",ELL_ERROR);
							downloadStagingArea->multi_free(1u, &address, &colorBufferBytesize, nullptr);
							continue;
						}
						if (downloadStagingArea->needsManualFlushOrInvalidate())
							driver->invalidateMappedMemoryRanges({{downloadStagingArea->getBuffer()->getBoundMemory(),address,colorBufferBytesize}});
					}
				}

				// create image view
				ICPUImageView::SCreationParams imgViewParams;
				imgViewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
				imgViewParams.format = image->getCreationParameters().format;
				imgViewParams.image = std::move(image);
				imgViewParams.viewType = ICPUImageView::ET_2D;
				imgViewParams.subresourceRange = {static_cast<IImage::E_ASPECT_FLAGS>(0u),0u,1u,0u,1u};
				imageView = ICPUImageView::create(std::move(imgViewParams));
			}

			// save as .EXR image
			{
				IAssetWriter::SAssetWriteParams wp(imageView.get());
				am->writeAsset(outputFileBundle[i].value().c_str(), wp);
			}

			auto getConvertedImageView = [&](core::smart_refctd_ptr<ICPUImage> image, const E_FORMAT& outFormat)
			{
				using CONVERSION_FILTER = CConvertFormatImageFilter<EF_UNKNOWN, EF_UNKNOWN, false, true, asset::CPrecomputedDither>;

				core::smart_refctd_ptr<ICPUImage> newConvertedImage;
				{
					auto referenceImageParams = image->getCreationParameters();
					auto referenceBuffer = image->getBuffer();
					auto referenceRegions = image->getRegions();
					auto referenceRegion = referenceRegions.begin();
					const auto newTexelOrBlockByteSize = asset::getTexelOrBlockBytesize(outFormat);

					auto newImageParams = referenceImageParams;
					auto newCpuBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(referenceRegion->getExtent().width * referenceRegion->getExtent().height * referenceRegion->getExtent().depth * newTexelOrBlockByteSize);
					auto newRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1);

					*newRegions->begin() = *referenceRegion;

					newImageParams.format = outFormat;
					newConvertedImage = ICPUImage::create(std::move(newImageParams));
					newConvertedImage->setBufferAndRegions(std::move(newCpuBuffer), newRegions);

					CONVERSION_FILTER convertFilter;
					CONVERSION_FILTER::state_type state;
					
					auto ditheringBundle = am->getAsset("../../media/blueNoiseDithering/LDR_RGBA.png", {});
					const auto ditheringStatus = ditheringBundle.getContents().empty();
					if (ditheringStatus)
					{
						os::Printer::log("ERROR (" + std::to_string(__LINE__) + " line): Could not load the dithering image!", ELL_ERROR);
						assert(ditheringStatus);
					}
					auto ditheringImage = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(ditheringBundle.getContents().begin()[0]);

					ICPUImageView::SCreationParams imageViewInfo;
					imageViewInfo.image = ditheringImage;
					imageViewInfo.format = ditheringImage->getCreationParameters().format;
					imageViewInfo.viewType = decltype(imageViewInfo.viewType)::ET_2D;
					imageViewInfo.components = {};
					imageViewInfo.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
					imageViewInfo.subresourceRange.baseArrayLayer = 0u;
					imageViewInfo.subresourceRange.baseMipLevel = 0u;
					imageViewInfo.subresourceRange.layerCount = ditheringImage->getCreationParameters().arrayLayers;
					imageViewInfo.subresourceRange.levelCount = ditheringImage->getCreationParameters().mipLevels;

					auto ditheringImageView = ICPUImageView::create(std::move(imageViewInfo));
					state.ditherState = _NBL_NEW(std::remove_pointer<decltype(state.ditherState)>::type, ditheringImageView.get());

					state.inImage = image.get();
					state.outImage = newConvertedImage.get();
					state.inOffset = { 0, 0, 0 };
					state.inBaseLayer = 0;
					state.outOffset = { 0, 0, 0 };
					state.outBaseLayer = 0;

					auto region = newConvertedImage->getRegions().begin();

					state.extent = region->getExtent();
					state.layerCount = region->imageSubresource.layerCount;
					state.inMipLevel = region->imageSubresource.mipLevel;
					state.outMipLevel = region->imageSubresource.mipLevel;

					if (!convertFilter.execute(&state))
						os::Printer::log("WARNING (" + std::to_string(__LINE__) + " line): Something went wrong while converting the image!", ELL_WARNING);

					_NBL_DELETE(state.ditherState);
				}

				// create image view
				ICPUImageView::SCreationParams imgViewParams;
				imgViewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
				imgViewParams.format = newConvertedImage->getCreationParameters().format;
				imgViewParams.image = std::move(newConvertedImage);
				imgViewParams.viewType = ICPUImageView::ET_2D;
				imgViewParams.subresourceRange = { static_cast<IImage::E_ASPECT_FLAGS>(0u),0u,1u,0u,1u };
				auto newImageView = ICPUImageView::create(std::move(imgViewParams));

				return newImageView;
			};

			// convert to EF_R8G8B8_SRGB and save it as .png and .jpg
			{
				auto newImageView = getConvertedImageView(imageView->getCreationParameters().image, EF_R8G8B8_SRGB);
				IAssetWriter::SAssetWriteParams wp(newImageView.get());
				std::string fileName = outputFileBundle[i].value().c_str();

				while (fileName.back() != '.')
					fileName.pop_back();

				const std::string& nonFormatFileName = fileName;
				am->writeAsset(nonFormatFileName + "png", wp);
				am->writeAsset(nonFormatFileName + "jpg", wp);
			}

			// destroy link to CPUBuffer's data (we need to free it)
			imageView->convertToDummyObject(~0u);

			// free the staging area allocation (no fence, we've already waited on it)
			downloadStagingArea->multi_free(1u,&address,&colorBufferBytesize,nullptr);

			// destroy image (implicit)

			//
			driver->endScene();
		}
	}

	return 0;
}