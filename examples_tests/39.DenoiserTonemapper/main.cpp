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
constexpr uint32_t calcDenoiserBuffersNeeded(E_IMAGE_INPUT denoiserType)
{
	return 4u+denoiserType;
}

using FFTClass = ext::FFT::FFT;

struct ImageToDenoise
{
	FFTClass::Parameters_t fftPushConstants[3];
	FFTClass::DispatchInfo_t fftDispatchInfo[3];
	core::smart_refctd_ptr<asset::ICPUImage> image[EII_COUNT] = { nullptr,nullptr,nullptr };
	core::smart_refctd_ptr<asset::ICPUImage> kernel = nullptr;
	uint32_t width = 0u, height = 0u;
	uint32_t colorTexelSize = 0u;
	E_IMAGE_INPUT denoiserType = EII_COUNT;
	VkExtent3D scaledKernelExtent;
	float bloomIntensity;
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
constexpr uint32_t tileWidth = 1024, tileHeight = 1024;
constexpr uint32_t denoiseTileDims[] = { tileWidth ,tileHeight };
constexpr uint32_t denoiseTileDimsWithOverlap[] = { tileWidth+overlap*2,tileHeight+overlap*2 };

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
	// TODO: this is a temporary fix for a problem solved in the Vulkan Branch
	params.StreamingUploadBufferSize = 1024*1024*1024; // for Color + 2 AoV of 8k images
	params.StreamingDownloadBufferSize = core::roundUp(params.StreamingUploadBufferSize/3u,256u); // for output image
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
		{
			os::Printer::log("Guess input from Commandline arguments",ELL_INFORMATION);
			for (auto i = 1ul; i < argc; ++i)
				arguments.emplace_back(argv[i]);
		}
		else
		{
			os::Printer::log("No arguments provided, running demo mode from ../exampleInputArguments.txt", ELL_INFORMATION);
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


	using LumaMeterClass = ext::LumaMeter::CLumaMeter;
	using ToneMapperClass = ext::ToneMapper::CToneMapper;

	constexpr uint32_t kComputeWGSize = FFTClass::DEFAULT_WORK_GROUP_SIZE; // if it changes, maybe it breaks stuff
	constexpr uint32_t colorChannelsFFT = 3u;
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

	constexpr uint32_t kernelSetDescCount = 4u;
	constexpr auto SharedDescriptorSetDescCount = 5u;
	core::smart_refctd_ptr<IGPUDescriptorSetLayout> kernelDescriptorSetLayout,sharedDescriptorSetLayout;
	core::smart_refctd_ptr<IGPUPipelineLayout> kernelPipelineLayout,sharedPipelineLayout;
	core::smart_refctd_ptr<IGPUComputePipeline> firstKernelFFTPipeline,lastKernelFFTPipeline,kernelNormalizationPipeline,
		deinterleavePipeline,intensityPipeline,
		secondLumaMeterAndFirstFFTPipeline,convolvePipeline,interleaveAndLastFFTPipeline;
	// Normalization of FFT spectrum
	struct NormalizationPushConstants
	{
		ext::FFT::uvec4 stride;
		uint32_t bitreverse_shift[2];
		float bloomIntensity;
	};
	{
		auto firstKernelFFTShader = driver->createShader(core::make_smart_refctd_ptr<ICPUShader>(R"===(
#version 450 core
#define _NBL_GLSL_WORKGROUP_SIZE_ 256
layout(local_size_x=_NBL_GLSL_WORKGROUP_SIZE_, local_size_y=1, local_size_z=1) in;

// kinda bad overdeclaration but oh well
#define _NBL_GLSL_EXT_FFT_MAX_DIM_SIZE_ 16384

// Input Descriptor
layout(set=0, binding=0) uniform sampler2D inputImage;
#define _NBL_GLSL_EXT_FFT_INPUT_DESCRIPTOR_DEFINED_

#include "nbl/builtin/glsl/ext/FFT/parameters_struct.glsl"
#include "nbl/builtin/glsl/ext/FFT/parameters.glsl"

#include <nbl/builtin/glsl/math/complex.glsl>
nbl_glsl_complex nbl_glsl_ext_FFT_getPaddedData(in ivec3 coordinate, in uint channel) 
{
	const vec2 inputSize = vec2(nbl_glsl_ext_FFT_Parameters_t_getDimensions().xy);
	const vec2 halfInputSize = inputSize*0.5;
	const vec2 relativeCoords = vec2(coordinate.xy)-halfInputSize;
	const vec2 inputSizeRcp = vec2(1.0)/inputSize;
    const vec4 texelValue = textureGrad(inputImage,(relativeCoords+vec2(0.5))*inputSizeRcp+vec2(0.5),vec2(inputSizeRcp.x,0.0),vec2(0.0,inputSizeRcp.y));
	return nbl_glsl_complex(texelValue[channel], 0.0f);
}
#define _NBL_GLSL_EXT_FFT_GET_PADDED_DATA_DEFINED_

#include "nbl/builtin/glsl/ext/FFT/default_compute_fft.comp"
		)==="));
		auto lastKernelFFTShader = driver->createShader(core::make_smart_refctd_ptr<ICPUShader>(R"===(
#version 450 core
#define _NBL_GLSL_WORKGROUP_SIZE_ 256
layout(local_size_x=_NBL_GLSL_WORKGROUP_SIZE_, local_size_y=1, local_size_z=1) in;

// kinda bad overdeclaration but oh well
#define _NBL_GLSL_EXT_FFT_MAX_DIM_SIZE_ 16384
#include <nbl/builtin/glsl/ext/FFT/types.glsl>

layout(set=0, binding=1) readonly restrict buffer InputBuffer
{
	nbl_glsl_ext_FFT_storage_t inData[];
};
#define _NBL_GLSL_EXT_FFT_INPUT_DESCRIPTOR_DEFINED_

layout(set=0, binding=2) writeonly restrict buffer OutputBuffer
{
	nbl_glsl_ext_FFT_storage_t outData[];
};
#define _NBL_GLSL_EXT_FFT_OUTPUT_DESCRIPTOR_DEFINED_

#include "nbl/builtin/glsl/ext/FFT/default_compute_fft.comp"
		)==="));
		auto kernelNormalizationShader = driver->createShader(core::make_smart_refctd_ptr<ICPUShader>(R"===(
#version 450 core
layout(local_size_x=16, local_size_y=16, local_size_z=1) in;

#include <nbl/builtin/glsl/ext/FFT/types.glsl>

layout(set=0, binding=2) readonly restrict buffer InputBuffer
{
	nbl_glsl_ext_FFT_storage_t inData[];
};
layout(set=0, binding=3, rg32f) uniform image2D NormalizedKernel[3];

layout(push_constant) uniform PushConstants
{
	uvec4 strides;
	uvec2 bitreverse_shift;
	float bloomIntensity;
} pc;

#include <nbl/builtin/glsl/colorspace/encodeCIEXYZ.glsl>

void main()
{
	nbl_glsl_complex value = inData[nbl_glsl_dot(gl_GlobalInvocationID,pc.strides.xyz)];
	
	// imaginary component will be 0, image shall be positive
	vec3 avg;
	for (uint i=0u; i<3u; i++)
		avg[i] = inData[pc.strides.z*i].x;
	const float power = (nbl_glsl_scRGBtoXYZ*avg).y;

	const uvec2 coord = bitfieldReverse(gl_GlobalInvocationID.xy)>>pc.bitreverse_shift;
	const nbl_glsl_complex shift = nbl_glsl_expImaginary(-nbl_glsl_PI*float(coord.x+coord.y));
	value = nbl_glsl_complex_mul(value,shift)/power;
	value = value*pc.bloomIntensity+nbl_glsl_complex(1.0-pc.bloomIntensity,0.0);
	imageStore(NormalizedKernel[gl_WorkGroupID.z],ivec2(coord),vec4(value,0.0,0.0));
}
		)==="));
		auto firstKernelFFTSpecializedShader = driver->createSpecializedShader(firstKernelFFTShader.get(),IGPUSpecializedShader::SInfo(nullptr,nullptr,"main",ISpecializedShader::ESS_COMPUTE));
		auto lastKernelFFTSpecializedShader = driver->createSpecializedShader(lastKernelFFTShader.get(),IGPUSpecializedShader::SInfo(nullptr,nullptr,"main",ISpecializedShader::ESS_COMPUTE));
		auto kernelNormalizationSpecializedShader = driver->createSpecializedShader(kernelNormalizationShader.get(),IGPUSpecializedShader::SInfo(nullptr,nullptr,"main",ISpecializedShader::ESS_COMPUTE));

		{
			IGPUSampler::SParams params =
			{
				{
					ISampler::ETC_CLAMP_TO_BORDER,
					ISampler::ETC_CLAMP_TO_BORDER,
					ISampler::ETC_CLAMP_TO_BORDER,
					ISampler::ETBC_FLOAT_OPAQUE_BLACK,
					ISampler::ETF_LINEAR,
					ISampler::ETF_LINEAR,
					ISampler::ESMM_LINEAR,
					0u,
					0u,
					ISampler::ECO_ALWAYS
				}
			};
			auto sampler = driver->createSampler(std::move(params));
			IGPUDescriptorSetLayout::SBinding binding[kernelSetDescCount] = {
				{0u,EDT_COMBINED_IMAGE_SAMPLER,1u,IGPUSpecializedShader::ESS_COMPUTE,&sampler},
				{1u,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
				{2u,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
				{3u,EDT_STORAGE_IMAGE,colorChannelsFFT,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
			};
			kernelDescriptorSetLayout = driver->createDescriptorSetLayout(binding,binding+kernelSetDescCount);
		}

		{
			SPushConstantRange pcRange[1] = {IGPUSpecializedShader::ESS_COMPUTE,0u,core::max(sizeof(FFTClass::Parameters_t),sizeof(NormalizationPushConstants))};
			kernelPipelineLayout = driver->createPipelineLayout(pcRange,pcRange+1u,core::smart_refctd_ptr(kernelDescriptorSetLayout));
		}

		firstKernelFFTPipeline = driver->createComputePipeline(nullptr,core::smart_refctd_ptr(kernelPipelineLayout),std::move(firstKernelFFTSpecializedShader));
		lastKernelFFTPipeline = driver->createComputePipeline(nullptr,core::smart_refctd_ptr(kernelPipelineLayout),std::move(lastKernelFFTSpecializedShader));
		kernelNormalizationPipeline = driver->createComputePipeline(nullptr,core::smart_refctd_ptr(kernelPipelineLayout),std::move(kernelNormalizationSpecializedShader));


		auto deinterleaveShader = driver->createShader(core::make_smart_refctd_ptr<ICPUShader>(R"===(
#version 450 core
#extension GL_EXT_shader_16bit_storage : require
#define _NBL_GLSL_EXT_LUMA_METER_FIRST_PASS_DEFINED_
#include "../ShaderCommon.glsl"
layout(binding = 0, std430) restrict readonly buffer ImageInputBuffer
{
	f16vec4 data[];
} inBuffers[EII_COUNT];
layout(binding = 1, std430) restrict writeonly buffer ImageOutputBuffer
{
	f16vec3_packed data[];
} outBuffers[EII_COUNT];
vec3 fetchData(in uvec3 texCoord)
{
	vec3 data = vec4(inBuffers[texCoord.z].data[texCoord.y*pc.data.inImageTexelPitch[texCoord.z]+texCoord.x]).xyz;
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
		barrier();
	}
	const uint addr = gl_GlobalInvocationID.y*pc.data.imageWidth+gl_GlobalInvocationID.x;
	outBuffers[gl_GlobalInvocationID.z].data[addr].x = float16_t(globalPixelData.x);
	outBuffers[gl_GlobalInvocationID.z].data[addr].y = float16_t(globalPixelData.y);
	outBuffers[gl_GlobalInvocationID.z].data[addr].z = float16_t(globalPixelData.z);
}
		)==="));
		auto intensityShader = driver->createShader(core::make_smart_refctd_ptr<ICPUShader>(R"===(
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
	return int((~pc.data.flags)&0x1u);
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

	float optixIntensity = 1.0;
	if (bool(pc.data.flags&0x2u))
	{
		nbl_glsl_ext_LumaMeter_PassInfo_t lumaPassInfo;
		lumaPassInfo.percentileRange[0] = pc.data.percentileRange[0];
		lumaPassInfo.percentileRange[1] = pc.data.percentileRange[1];
		float measuredLumaLog2 = nbl_glsl_ext_LumaMeter_getMeasuredLumaLog2(nbl_glsl_ext_ToneMapper_getLumaMeterOutput(),lumaPassInfo);
		if (firstInvocation)
		{
			const bool beforeDenoise = bool(pc.data.flags&0x1u);
			measuredLumaLog2 += beforeDenoise ? pc.data.denoiserExposureBias:0.0;
			optixIntensity = nbl_glsl_ext_LumaMeter_getOptiXIntensity(measuredLumaLog2);
		}
	}
	
	if (firstInvocation)
		intensity[pc.data.intensityBufferDWORDOffset] = optixIntensity;
}
		)==="));
		auto secondLumaMeterAndFirstFFTShader = driver->createShader(core::make_smart_refctd_ptr<ICPUShader>(R"===(
#version 450 core
#extension GL_EXT_shader_16bit_storage : require
#define _NBL_GLSL_EXT_LUMA_METER_FIRST_PASS_DEFINED_
#include "../ShaderCommon.glsl"
layout(binding = 0, std430) restrict readonly buffer ImageInputBuffer
{
	f16vec3_packed inBuffer[];
};
#define _NBL_GLSL_EXT_FFT_INPUT_DESCRIPTOR_DEFINED_
layout(binding = 1, std430) restrict writeonly buffer SpectrumOutputBuffer
{
	vec2 outSpectrum[];
};
#define _NBL_GLSL_EXT_FFT_OUTPUT_DESCRIPTOR_DEFINED_



#include <nbl/builtin/glsl/math/complex.glsl>
nbl_glsl_complex nbl_glsl_ext_FFT_getPaddedData(ivec3 coordinate, in uint channel);
#define _NBL_GLSL_EXT_FFT_GET_PADDED_DATA_DEFINED_


uvec3 nbl_glsl_ext_FFT_Parameters_t_getDimensions()
{
	return uvec3(pc.data.imageWidth,pc.data.imageHeight,1u);
}
uint nbl_glsl_ext_FFT_Parameters_t_getLog2FFTSize()
{
	return CommonPushConstants_getPassLog2FFTSize(0);
}
bool nbl_glsl_ext_FFT_Parameters_t_getIsInverse()
{
	return false;
}
uint nbl_glsl_ext_FFT_Parameters_t_getDirection()
{
	return 0u;
}
#define _NBL_GLSL_EXT_FFT_PARAMETERS_METHODS_DECLARED_


void nbl_glsl_ext_FFT_setData(in uvec3 coordinate, in uint channel, in nbl_glsl_complex complex_value)
{
	const uint index = ((channel<<CommonPushConstants_getPassLog2FFTSize(0))+coordinate.x)*pc.data.imageHeight+coordinate.y;
	outSpectrum[index] = complex_value;
}
#define _NBL_GLSL_EXT_FFT_SET_DATA_DEFINED_


#define _NBL_GLSL_EXT_FFT_MAIN_DEFINED_
#include "nbl/builtin/glsl/ext/FFT/default_compute_fft.comp"


float scaledLogLuma;
nbl_glsl_complex nbl_glsl_ext_FFT_getPaddedData(ivec3 coordinate, in uint channel) 
{
	ivec3 oldCoord = coordinate;
	nbl_glsl_ext_FFT_wrap_coord(coordinate);

	const uint index = coordinate.y*pc.data.imageWidth+coordinate.x;

	// rewrite this fetch at some point
	nbl_glsl_complex retval; retval.y = 0.0;
	switch (channel)
	{
		case 2u:
			retval[0] = float(inBuffer[index].z);
			break;
		case 1u:
			retval[0] = float(inBuffer[index].y);
			break;
		default:
			scaledLogLuma += nbl_glsl_ext_LumaMeter_local_process(all(equal(coordinate,oldCoord)),vec3(inBuffer[index].x,inBuffer[index].y,inBuffer[index].z));
			retval[0] = float(inBuffer[index].x);
			break;
	}
	return retval;
}

void main()
{
	#if _NBL_GLSL_EXT_LUMA_METER_MODE_DEFINED_==_NBL_GLSL_EXT_LUMA_METER_MODE_MEDIAN
		nbl_glsl_ext_LumaMeter_clearHistogram();
	#endif
	nbl_glsl_ext_LumaMeter_clearFirstPassOutput();


	// Virtual Threads Calculation
	const uint log2FFTSize = nbl_glsl_ext_FFT_Parameters_t_getLog2FFTSize();
	const uint item_per_thread_count = 0x1u<<(log2FFTSize-_NBL_GLSL_WORKGROUP_SIZE_LOG2_);
	for(uint channel=0u; channel<3u; channel++)
	{
		scaledLogLuma = 0.f;
		// Load Values into local memory
		for(uint t=0u; t<item_per_thread_count; t++)
		{
			const uint tid = (t<<_NBL_GLSL_WORKGROUP_SIZE_LOG2_)|gl_LocalInvocationIndex;
			const uint trueDim = nbl_glsl_ext_FFT_Parameters_t_getDimensions()[nbl_glsl_ext_FFT_Parameters_t_getDirection()];
			nbl_glsl_ext_FFT_impl_values[t] = nbl_glsl_ext_FFT_getPaddedData(nbl_glsl_ext_FFT_getPaddedCoordinates(tid,log2FFTSize,trueDim),channel);
		}
		if (channel==0u)
		{
			nbl_glsl_ext_LumaMeter_setFirstPassOutput(nbl_glsl_ext_LumaMeter_workgroup_process(scaledLogLuma));
			// prevent overlap between different usages of shared memory
			barrier();
		}
		// do FFT
		nbl_glsl_ext_FFT_preloaded(false,log2FFTSize);
		// write out to main memory
		for(uint t=0u; t<item_per_thread_count; t++)
		{
			const uint tid = (t<<_NBL_GLSL_WORKGROUP_SIZE_LOG2_)|gl_LocalInvocationIndex;
			nbl_glsl_ext_FFT_setData(nbl_glsl_ext_FFT_getCoordinates(tid),channel,nbl_glsl_ext_FFT_impl_values[t]);
		}
	}
}
		)==="));
		auto convolveShader = driver->createShader(core::make_smart_refctd_ptr<ICPUShader>(R"===(
#version 450 core
#extension GL_EXT_shader_16bit_storage : require

// nasty and ugly but oh well
#define _NBL_GLSL_SCRATCH_SHARED_DEFINED_ sharedScratch
#define _NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_ 1024
shared uint _NBL_GLSL_SCRATCH_SHARED_DEFINED_[_NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_];

#include "../ShaderCommon.glsl"
layout(binding = 1, std430) restrict buffer SpectrumBuffer
{
	vec2 spectrum[];
};
#define _NBL_GLSL_EXT_FFT_INPUT_DESCRIPTOR_DEFINED_
#define _NBL_GLSL_EXT_FFT_OUTPUT_DESCRIPTOR_DEFINED_


layout(binding=4) uniform sampler2D NormalizedKernel[3];


#include <nbl/builtin/glsl/math/complex.glsl>


uvec3 nbl_glsl_ext_FFT_Parameters_t_getDimensions()
{
	return uvec3(0x1u<<CommonPushConstants_getPassLog2FFTSize(0),pc.data.imageHeight,1u);
}
uint nbl_glsl_ext_FFT_Parameters_t_getLog2FFTSize()
{
	return CommonPushConstants_getPassLog2FFTSize(1);
}
bool nbl_glsl_ext_FFT_Parameters_t_getIsInverse()
{
	return bool(0xdeadbeefu);
}
uint nbl_glsl_ext_FFT_Parameters_t_getDirection()
{
	return 1u;
}
#define _NBL_GLSL_EXT_FFT_PARAMETERS_METHODS_DECLARED_


nbl_glsl_complex nbl_glsl_ext_FFT_getPaddedData(ivec3 coordinate, in uint channel);
#define _NBL_GLSL_EXT_FFT_GET_PADDED_DATA_DEFINED_
void nbl_glsl_ext_FFT_setData(in uvec3 coordinate, in uint channel, in nbl_glsl_complex complex_value)
{
	const uint index = ((channel<<CommonPushConstants_getPassLog2FFTSize(0))+coordinate.x)*pc.data.imageHeight+coordinate.y;
	spectrum[index] = complex_value;
}
#define _NBL_GLSL_EXT_FFT_SET_DATA_DEFINED_

#define _NBL_GLSL_EXT_FFT_MAIN_DEFINED_
#include "nbl/builtin/glsl/ext/FFT/default_compute_fft.comp"

void convolve(in uint item_per_thread_count, in uint ch) 
{
	for(uint t=0u; t<item_per_thread_count; t++)
	{
		const uint tid = _NBL_GLSL_WORKGROUP_SIZE_*t+gl_LocalInvocationIndex;

		nbl_glsl_complex sourceSpectrum = nbl_glsl_ext_FFT_impl_values[t];
		
		//
		const uvec3 coords = nbl_glsl_ext_FFT_getCoordinates(tid);
        vec2 uv = vec2(bitfieldReverse(coords.xy))/vec2(4294967296.f);

		uv += pc.data.kernel_half_pixel_size;
		//
		nbl_glsl_complex convSpectrum = textureLod(NormalizedKernel[ch],uv,0).xy;
		nbl_glsl_ext_FFT_impl_values[t] = nbl_glsl_complex_mul(sourceSpectrum,convSpectrum);
	}
}

void main()
{
	// Virtual Threads Calculation
	const uint log2FFTSize = nbl_glsl_ext_FFT_Parameters_t_getLog2FFTSize();
	const uint item_per_thread_count = 0x1u<<(log2FFTSize-_NBL_GLSL_WORKGROUP_SIZE_LOG2_);
	for(uint channel=0u; channel<3u; channel++)
	{
		// Load Values into local memory
		for(uint t=0u; t<item_per_thread_count; t++)
		{
			const uint tid = (t<<_NBL_GLSL_WORKGROUP_SIZE_LOG2_)|gl_LocalInvocationIndex;
			const uint trueDim = nbl_glsl_ext_FFT_Parameters_t_getDimensions()[nbl_glsl_ext_FFT_Parameters_t_getDirection()];
			nbl_glsl_ext_FFT_impl_values[t] = nbl_glsl_ext_FFT_getPaddedData(nbl_glsl_ext_FFT_getPaddedCoordinates(tid,log2FFTSize,trueDim),channel);
		}
		nbl_glsl_ext_FFT_preloaded(false,log2FFTSize);
		barrier();

		convolve(item_per_thread_count,channel);
	
		barrier();
		nbl_glsl_ext_FFT_preloaded(true,log2FFTSize);
		// write out to main memory
		for(uint t=0u; t<item_per_thread_count; t++)
		{
			const uint tid = (t<<_NBL_GLSL_WORKGROUP_SIZE_LOG2_)|gl_LocalInvocationIndex;
			const uint trueDim = nbl_glsl_ext_FFT_Parameters_t_getDimensions()[nbl_glsl_ext_FFT_Parameters_t_getDirection()];
			// we also prevent certain threads from writing the memory out
			const uint padding = ((0x1u<<log2FFTSize)-trueDim)>>1u;
			const uint shifted = tid-padding;
			if (tid>=padding && shifted<trueDim)
				nbl_glsl_ext_FFT_setData(ivec3(nbl_glsl_ext_FFT_getCoordinates(shifted)),channel,nbl_glsl_ext_FFT_impl_values[t]);
		}
	}
}

nbl_glsl_complex nbl_glsl_ext_FFT_getPaddedData(ivec3 coordinate, in uint channel) 
{
	if (!nbl_glsl_ext_FFT_wrap_coord(coordinate))
		return nbl_glsl_complex(0.f,0.f);
	const uint index = ((channel<<CommonPushConstants_getPassLog2FFTSize(0))+coordinate.x)*pc.data.imageHeight+coordinate.y;
	return spectrum[index];
}
		)==="));
		auto interleaveAndLastFFTShader = driver->createShader(core::make_smart_refctd_ptr<ICPUShader>(R"===(
#version 450 core
#extension GL_EXT_shader_16bit_storage : require

// nasty and ugly but oh well
#define _NBL_GLSL_SCRATCH_SHARED_DEFINED_ sharedScratch
#define _NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_ 1024
shared uint _NBL_GLSL_SCRATCH_SHARED_DEFINED_[_NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_];


#include "../ShaderCommon.glsl"
#include "nbl/builtin/glsl/ext/ToneMapper/operators.glsl"
layout(binding = 0, std430) restrict buffer ImageOutputBuffer
{
	f16vec4 outBuffer[];
};
#define _NBL_GLSL_EXT_FFT_OUTPUT_DESCRIPTOR_DEFINED_
layout(binding = 1, std430) restrict readonly buffer SpectrumInputBuffer
{
	vec2 inSpectrum[];
};
#define _NBL_GLSL_EXT_FFT_INPUT_DESCRIPTOR_DEFINED_
layout(binding = 3, std430) restrict readonly buffer IntensityBuffer
{
	float intensity[];
};


#include <nbl/builtin/glsl/math/complex.glsl>
nbl_glsl_complex nbl_glsl_ext_FFT_getPaddedData(ivec3 coordinate, in uint channel);
#define _NBL_GLSL_EXT_FFT_GET_PADDED_DATA_DEFINED_

uvec3 nbl_glsl_ext_FFT_Parameters_t_getDimensions()
{
	return uvec3(0x1u<<CommonPushConstants_getPassLog2FFTSize(0),pc.data.imageHeight,1u);
}
uint nbl_glsl_ext_FFT_Parameters_t_getLog2FFTSize()
{
	return CommonPushConstants_getPassLog2FFTSize(0);
}
bool nbl_glsl_ext_FFT_Parameters_t_getIsInverse()
{
	return true;
}
uint nbl_glsl_ext_FFT_Parameters_t_getDirection()
{
	return 0u;
}
#define _NBL_GLSL_EXT_FFT_PARAMETERS_METHODS_DECLARED_


void nbl_glsl_ext_FFT_setData(in uvec3 coordinate, in uint channel, in nbl_glsl_complex complex_value)
{
	ivec2 coords = ivec2(coordinate.xy);
	const uint padding_size = (0x1u<<nbl_glsl_ext_FFT_Parameters_t_getLog2FFTSize())-pc.data.imageWidth;
	coords.x -= int(padding_size>>1u);
    if (coords.x<0 || coords.x>=int(pc.data.imageWidth))
		return;
	
	uint dataOffset = coords.y*pc.data.inImageTexelPitch[EII_COLOR]+coords.x;	
	vec3 color = vec4(outBuffer[dataOffset]).xyz;
	color[channel] = complex_value.x;
	if (channel==nbl_glsl_ext_FFT_Parameters_t_getMaxChannel())
	{
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
	}
	outBuffer[dataOffset] = f16vec4(vec4(color,1.f));
}
#define _NBL_GLSL_EXT_FFT_SET_DATA_DEFINED_


#define _NBL_GLSL_EXT_FFT_MAIN_DEFINED_
#include "nbl/builtin/glsl/ext/FFT/default_compute_fft.comp"


void main()
{
	// Virtual Threads Calculation
	const uint log2FFTSize = nbl_glsl_ext_FFT_Parameters_t_getLog2FFTSize();
	const uint item_per_thread_count = 0x1u<<(log2FFTSize-_NBL_GLSL_WORKGROUP_SIZE_LOG2_);
	for(uint channel=0u; channel<3u; channel++)
	{
		// Load Values into local memory
		for(uint t=0u; t<item_per_thread_count; t++)
		{
			const uint tid = (t<<_NBL_GLSL_WORKGROUP_SIZE_LOG2_)|gl_LocalInvocationIndex;
			const uint trueDim = nbl_glsl_ext_FFT_Parameters_t_getDimensions()[nbl_glsl_ext_FFT_Parameters_t_getDirection()];
			nbl_glsl_ext_FFT_impl_values[t] = nbl_glsl_ext_FFT_getPaddedData(nbl_glsl_ext_FFT_getPaddedCoordinates(tid,log2FFTSize,trueDim),channel);
		}
		// do FFT
		nbl_glsl_ext_FFT_preloaded(true,log2FFTSize);
		// write out to main memory
		for(uint t=0u; t<item_per_thread_count; t++)
		{
			const uint tid = (t<<_NBL_GLSL_WORKGROUP_SIZE_LOG2_)|gl_LocalInvocationIndex;
			nbl_glsl_ext_FFT_setData(nbl_glsl_ext_FFT_getCoordinates(tid),channel,nbl_glsl_ext_FFT_impl_values[t]);
		}
	}
}

nbl_glsl_complex nbl_glsl_ext_FFT_getPaddedData(ivec3 coordinate, in uint channel) 
{
	if (!nbl_glsl_ext_FFT_wrap_coord(coordinate))
		return nbl_glsl_complex(0.f,0.f);
	const uint index = ((channel<<CommonPushConstants_getPassLog2FFTSize(0))+coordinate.x)*pc.data.imageHeight+coordinate.y;
	return inSpectrum[index];
}
		)==="));
		struct SpecializationConstants
		{
			uint32_t workgroupSize = kComputeWGSize;
			uint32_t enumEII_COLOR = EII_COLOR;
			uint32_t enumEII_ALBEDO = EII_ALBEDO;
			uint32_t enumEII_NORMAL = EII_NORMAL;
			uint32_t enumEII_COUNT = EII_COUNT;
		} specData;
		auto specConstantBuffer = core::make_smart_refctd_ptr<CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t> > >(sizeof(SpecializationConstants), &specData, core::adopt_memory);
		IGPUSpecializedShader::SInfo specInfo = {	core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IGPUSpecializedShader::SInfo::SMapEntry> >
													(
														std::initializer_list<IGPUSpecializedShader::SInfo::SMapEntry>
														{
															{0u,offsetof(SpecializationConstants,workgroupSize),sizeof(SpecializationConstants::workgroupSize)},
															{1u,offsetof(SpecializationConstants,enumEII_COLOR),sizeof(SpecializationConstants::enumEII_COLOR)},
															{2u,offsetof(SpecializationConstants,enumEII_ALBEDO),sizeof(SpecializationConstants::enumEII_ALBEDO)},
															{3u,offsetof(SpecializationConstants,enumEII_NORMAL),sizeof(SpecializationConstants::enumEII_NORMAL)},
															{4u,offsetof(SpecializationConstants,enumEII_COUNT),sizeof(SpecializationConstants::enumEII_COUNT)}
														}
													),
													core::smart_refctd_ptr(specConstantBuffer),"main",ISpecializedShader::ESS_COMPUTE
												};
		auto deinterleaveSpecializedShader = driver->createSpecializedShader(deinterleaveShader.get(),specInfo);
		auto intensitySpecializedShader = driver->createSpecializedShader(intensityShader.get(),specInfo);
		auto secondLumaMeterAndFirstFFTSpecializedShader = driver->createSpecializedShader(secondLumaMeterAndFirstFFTShader.get(),specInfo);
		auto convolveSpecializedShader = driver->createSpecializedShader(convolveShader.get(),specInfo);
		auto interleaveAndLastFFTSpecializedShader = driver->createSpecializedShader(interleaveAndLastFFTShader.get(),specInfo);

		{
			core::smart_refctd_ptr<IGPUSampler> samplers[colorChannelsFFT];
			{
				IGPUSampler::SParams params =
				{
					{
						ISampler::ETC_REPEAT,
						ISampler::ETC_REPEAT,
						ISampler::ETC_REPEAT,
						ISampler::ETBC_FLOAT_OPAQUE_BLACK,
						ISampler::ETF_LINEAR, // is it needed?
						ISampler::ETF_LINEAR,
						ISampler::ESMM_NEAREST,
						0u,
						0u,
						ISampler::ECO_ALWAYS
					}
				};
				auto sampler = driver->createSampler(std::move(params));
				std::fill_n(samplers,colorChannelsFFT,sampler);
			}
			IGPUDescriptorSetLayout::SBinding binding[SharedDescriptorSetDescCount] = {
				{0u,EDT_STORAGE_BUFFER,EII_COUNT,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
				{1u,EDT_STORAGE_BUFFER,EII_COUNT,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
				{2u,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
				{3u,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
				{4u,EDT_COMBINED_IMAGE_SAMPLER,colorChannelsFFT,IGPUSpecializedShader::ESS_COMPUTE,samplers}
			};
			sharedDescriptorSetLayout = driver->createDescriptorSetLayout(binding,binding+SharedDescriptorSetDescCount);
		}

		{
			SPushConstantRange pcRange[1] = {IGPUSpecializedShader::ESS_COMPUTE,0u,sizeof(CommonPushConstants)};
			sharedPipelineLayout = driver->createPipelineLayout(pcRange,pcRange+sizeof(pcRange)/sizeof(SPushConstantRange),core::smart_refctd_ptr(sharedDescriptorSetLayout));
		}

		deinterleavePipeline = driver->createComputePipeline(nullptr,core::smart_refctd_ptr(sharedPipelineLayout),std::move(deinterleaveSpecializedShader));
		intensityPipeline = driver->createComputePipeline(nullptr,core::smart_refctd_ptr(sharedPipelineLayout),std::move(intensitySpecializedShader));
		secondLumaMeterAndFirstFFTPipeline = driver->createComputePipeline(nullptr,core::smart_refctd_ptr(sharedPipelineLayout),std::move(secondLumaMeterAndFirstFFTSpecializedShader));
		convolvePipeline = driver->createComputePipeline(nullptr,core::smart_refctd_ptr(sharedPipelineLayout),std::move(convolveSpecializedShader));
		interleaveAndLastFFTPipeline = driver->createComputePipeline(nullptr,core::smart_refctd_ptr(sharedPipelineLayout),std::move(interleaveAndLastFFTSpecializedShader));
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
	const auto& bloomRelativeScaleBundle = cmdHandler.getBloomRelativeScaleBundle();
	const auto& bloomIntensityBundle = cmdHandler.getBloomIntensityBundle();
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
	uint32_t fftScratchSize = 0u;
	{
		asset::IAssetLoader::SAssetLoadParams lp(0ull,nullptr);
		auto default_kernel_image_bundle = am->getAsset("../../media/kernels/physical_flare_512.exr",lp); // TODO: make it a builtins?

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

				const float bloomRelativeScale = bloomRelativeScaleBundle[i].value();
				{
					auto kerDim = outParam.kernel->getCreationParameters().extent;
					float kernelScale,minKernelScale;
					if (extent.width<extent.height)
					{
						minKernelScale = 2.f/float(kerDim.width);
						kernelScale = float(extent.width)*bloomRelativeScale/float(kerDim.width);
					}
					else
					{
						minKernelScale = 2.f/float(kerDim.height);
						kernelScale = float(extent.height)*bloomRelativeScale/float(kerDim.height);
					}
					//
					if (kernelScale>1.f)
						os::Printer::log(imageIDString + "Bloom Kernel loose sharpness, increase resolution of bloom kernel or reduce its relative scale!", ELL_WARNING);
					else if (kernelScale<minKernelScale)
						os::Printer::log(imageIDString + "Bloom Kernel relative scale pathologically small, clamping to prevent division by 0!", ELL_WARNING);
					outParam.scaledKernelExtent.width = core::max(core::ceil(float(kerDim.width)*kernelScale),2u);
					outParam.scaledKernelExtent.height = core::max(core::ceil(float(kerDim.height)*kernelScale),2u);
					outParam.scaledKernelExtent.depth = 1u;
				}
				const auto marginSrcDim = [extent,outParam]() -> auto
				{
					auto tmp = extent;
					for (auto i=0u; i<3u; i++)
					{
						const auto coord = (&outParam.scaledKernelExtent.width)[i];
						if (coord>1u)
							(&tmp.width)[i] += coord-1u;
					}
					return tmp;
				}();
				fftScratchSize = core::max(FFTClass::getOutputBufferSize(usingHalfFloatFFTStorage,outParam.scaledKernelExtent,colorChannelsFFT)*2u,fftScratchSize);
				fftScratchSize = core::max(FFTClass::getOutputBufferSize(usingHalfFloatFFTStorage,marginSrcDim,colorChannelsFFT),fftScratchSize);
				// TODO: maybe move them to nested loop and compute JIT
				{
					auto* fftPushConstants = outParam.fftPushConstants;
					auto* fftDispatchInfo = outParam.fftDispatchInfo;
					const ISampler::E_TEXTURE_CLAMP fftPadding[2] = {ISampler::ETC_MIRROR,ISampler::ETC_MIRROR};
					const auto passes = FFTClass::buildParameters<false>(false,colorChannelsFFT,extent,fftPushConstants,fftDispatchInfo,fftPadding,marginSrcDim);
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

				outParam.bloomIntensity = bloomIntensityBundle[i].value();

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

	// keep all CUDA links in an array (less code to map/unmap)
	constexpr uint32_t kMaxDenoiserBuffers = calcDenoiserBuffersNeeded(EII_NORMAL);
	cuda::CCUDAHandler::GraphicsAPIObjLink<video::IGPUBuffer> bufferLinks[kMaxDenoiserBuffers];
	// set-up denoisers
	constexpr size_t IntensityValuesSize = sizeof(float);
	auto& intensityBuffer = bufferLinks[0];
	auto& denoiserState = bufferLinks[0];
	auto& scratch = bufferLinks[1];
	auto& temporaryPixelBuffer = bufferLinks[2];
	auto& colorPixelBuffer = bufferLinks[3];
	auto& albedoPixelBuffer = bufferLinks[4];
	auto& normalPixelBuffer = bufferLinks[5];
	//auto denoised;
	size_t denoiserStateBufferSize = 0ull;
	{
		size_t scratchBufferSize = fftScratchSize;
		size_t tempBufferSize = fftScratchSize;
		for (uint32_t i=0u; i<EII_COUNT; i++)
		{
			auto& denoiser = denoisers[i].m_denoiser;

			OptixDenoiserSizes m_denoiserMemReqs;
			if (denoiser->computeMemoryResources(&m_denoiserMemReqs, denoiseTileDims)!=OPTIX_SUCCESS)
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
			tempBufferSize = core::max(tempBufferSize,forcedOptiXFormatPixelStride*(i+1)*maxResolution[0]*maxResolution[1]);
		}
		std::string message = "Total VRAM consumption for Denoiser algorithm: ";
		os::Printer::log(message+std::to_string(denoiserStateBufferSize+scratchBufferSize+tempBufferSize), ELL_INFORMATION);

		if (check_error(tempBufferSize==0ull,"No input files at all!"))
			return error_code;

		denoiserState = driver->createDeviceLocalGPUBufferOnDedMem(denoiserStateBufferSize+IntensityValuesSize);
		if (check_error(!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::registerBuffer(&denoiserState)),"Could not register buffer for Denoiser states!"))
			return error_code;

		temporaryPixelBuffer = driver->createDeviceLocalGPUBufferOnDedMem(tempBufferSize);
		if (check_error(!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::registerBuffer(&temporaryPixelBuffer)),"Could not register buffer for Denoiser scratch memory!"))
			return error_code;
		scratch = driver->createDeviceLocalGPUBufferOnDedMem(scratchBufferSize);
		if (check_error(!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::registerBuffer(&scratch)), "Could not register buffer for Denoiser temporary memory with CUDA natively!"))
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
			shaderConstants.imageWidth = param.width;
			shaderConstants.imageHeight = param.height;

			assert(intensityBufferOffset%IntensityValuesSize==0u);
			shaderConstants.intensityBufferDWORDOffset = intensityBufferOffset/IntensityValuesSize;
			shaderConstants.denoiserExposureBias = denoiserExposureBiasBundle[i].value();

			assert(param.fftPushConstants[0].getLog2FFTSize()==param.fftPushConstants[2].getLog2FFTSize());
			shaderConstants.flags = (param.fftPushConstants[1].getLog2FFTSize()<<7u)|(param.fftPushConstants[0].getLog2FFTSize()<<2u)|0b11u; // (autoexposureOn<<1)|beforeDenoise
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
						shaderConstants.flags &= ~0b10u; // ~(autoexposureOn<<1)
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
		uint32_t inImageByteOffset[EII_COUNT];
		{
			asset::ICPUBuffer* buffersToUpload[EII_COUNT];
			size_t inputSize = 0u;
			for (uint32_t j=0u; j<denoiserInputCount; j++)
			{
				buffersToUpload[j] = param.image[j]->getBuffer();
				inputSize += buffersToUpload[j]->getSize();
			}
			if (inputSize>=params.StreamingUploadBufferSize)
			{
				printf("[ERROR] Denoiser Failed, input too large to fit in VRAM, Streaming Denoise not implemented yet!");
				return -1;
			}
			auto gpubuffers = driver->getGPUObjectsFromAssets(buffersToUpload,buffersToUpload+denoiserInputCount,&assetConverter);

			bool skip = false;
			auto createLinkAndRegister = [&makeImageIDString,i,&skip,&gpubuffers](auto ix) -> auto
			{
				cuda::CCUDAHandler::GraphicsAPIObjLink<IGPUBuffer> retval = core::smart_refctd_ptr<IGPUBuffer>(gpubuffers->operator[](ix)->getBuffer());
				if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::registerBuffer(&retval)))
				{
					os::Printer::log(makeImageIDString(i) + "Could not register the image data buffer with CUDA, skipping image!", ELL_ERROR);
					skip = true;
				}
				return retval;
			};
			colorPixelBuffer = createLinkAndRegister(EII_COLOR);
			if (denoiserInputCount>EII_ALBEDO)
				albedoPixelBuffer = createLinkAndRegister(EII_ALBEDO);
			if (denoiserInputCount>EII_NORMAL)
				normalPixelBuffer = createLinkAndRegister(EII_NORMAL);
			if (skip)
				continue;

			for (uint32_t j=0u; j<denoiserInputCount; j++)
			{
				auto offsetPair = gpubuffers->operator[](j);
				// make sure cache doesn't retain the GPU object paired to CPU object (could have used a custom IGPUObjectFromAssetConverter derived class with overrides to achieve this)
				am->removeCachedGPUObject(buffersToUpload[j],offsetPair);

				auto image = param.image[j];
				const auto& creationParameters = image->getCreationParameters();
				assert(asset::getTexelOrBlockBytesize(creationParameters.format)==param.colorTexelSize);
				// set up some image pitch and offset info
				shaderConstants.inImageTexelPitch[j] = image->getRegions().begin()[0].bufferRowLength;
				inImageByteOffset[j] = offsetPair->getOffset();
			}
		}

		// process
		{
			// get the bloom kernel FFT Spectrum
			core::smart_refctd_ptr<IGPUImageView> kernelNormalizedSpectrums[colorChannelsFFT];
			{
				// kernel inputs
				core::smart_refctd_ptr<IGPUImageView> kerImageView;
				{
					auto kerGpuImages = driver->getGPUObjectsFromAssets(&param.kernel, &param.kernel + 1u, &assetConverter);


					IGPUImageView::SCreationParams kerImgViewInfo;
					kerImgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
					kerImgViewInfo.image = kerGpuImages->operator[](0u);

					// make sure cache doesn't retain the GPU object paired to CPU object (could have used a custom IGPUObjectFromAssetConverter derived class with overrides to achieve this)
					am->removeCachedGPUObject(param.kernel.get(), kerImgViewInfo.image);

					kerImgViewInfo.viewType = IGPUImageView::ET_2D;
					kerImgViewInfo.format = kerImgViewInfo.image->getCreationParameters().format;
					kerImgViewInfo.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0u);
					kerImgViewInfo.subresourceRange.baseMipLevel = 0;
					kerImgViewInfo.subresourceRange.levelCount = kerImgViewInfo.image->getCreationParameters().mipLevels;
					kerImgViewInfo.subresourceRange.baseArrayLayer = 0;
					kerImgViewInfo.subresourceRange.layerCount = 1;
					kerImageView = driver->createImageView(std::move(kerImgViewInfo));
				}

				// kernel outputs
				auto paddedKernelExtent = FFTClass::padDimensions(param.scaledKernelExtent);
				for (uint32_t i=0u; i<colorChannelsFFT; i++)
				{
					video::IGPUImage::SCreationParams imageParams;
					imageParams.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
					imageParams.type = asset::IImage::ET_2D;
					imageParams.format = EF_R32G32_SFLOAT;
					imageParams.extent = {paddedKernelExtent.width,paddedKernelExtent.height,1u};
					imageParams.mipLevels = 1u;
					imageParams.arrayLayers = 1u;
					imageParams.samples = asset::IImage::ESCF_1_BIT;

					video::IGPUImageView::SCreationParams viewParams;
					viewParams.flags = static_cast<video::IGPUImageView::E_CREATE_FLAGS>(0u);
					viewParams.image = driver->createGPUImageOnDedMem(std::move(imageParams),driver->getDeviceLocalGPUMemoryReqs());
					viewParams.viewType = video::IGPUImageView::ET_2D;
					viewParams.format = EF_R32G32_SFLOAT;
					viewParams.components = {};
					viewParams.subresourceRange = {};
					viewParams.subresourceRange.levelCount = 1u;
					viewParams.subresourceRange.layerCount = 1u;
					kernelNormalizedSpectrums[i] = driver->createImageView(std::move(viewParams));
				}

				//
				FFTClass::Parameters_t fftPushConstants[2];
				FFTClass::DispatchInfo_t fftDispatchInfo[2];
				const ISampler::E_TEXTURE_CLAMP fftPadding[2] = { ISampler::ETC_CLAMP_TO_BORDER,ISampler::ETC_CLAMP_TO_BORDER };
				const auto passes = FFTClass::buildParameters(false,colorChannelsFFT,param.scaledKernelExtent,fftPushConstants,fftDispatchInfo,fftPadding);

				// the kernel's FFTs
				{
					auto kernelDescriptorSet = driver->createDescriptorSet(core::smart_refctd_ptr(kernelDescriptorSetLayout));
					{
						IGPUDescriptorSet::SDescriptorInfo infos[kernelSetDescCount+colorChannelsFFT-1u];
						infos[0].desc = kerImageView;
						infos[0].image.sampler = nullptr; // immutable
						infos[1].desc = core::smart_refctd_ptr<IGPUBuffer>(temporaryPixelBuffer.getObject());
						infos[1].buffer = {0u,fftScratchSize>>1u};
						infos[2].desc = core::smart_refctd_ptr<IGPUBuffer>(temporaryPixelBuffer.getObject());
						infos[2].buffer = {fftScratchSize>>1u,fftScratchSize};
						for (uint32_t i=0u; i<colorChannelsFFT; i++)
						{
							infos[3+i].desc = kernelNormalizedSpectrums[i];
							infos[3+i].image.sampler = nullptr; // storage
						}
						IGPUDescriptorSet::SWriteDescriptorSet writes[kernelSetDescCount] =
						{
							{kernelDescriptorSet.get(),0u,0u,1u,EDT_COMBINED_IMAGE_SAMPLER,infos+0u},
							{kernelDescriptorSet.get(),1u,0u,1u,EDT_STORAGE_BUFFER,infos+1u},
							{kernelDescriptorSet.get(),2u,0u,1u,EDT_STORAGE_BUFFER,infos+2u},
							{kernelDescriptorSet.get(),3u,0u,colorChannelsFFT,EDT_STORAGE_IMAGE,infos+3u}
						};
						driver->updateDescriptorSets(kernelSetDescCount,writes,0u,nullptr);
					}
					driver->bindDescriptorSets(EPBP_COMPUTE,kernelPipelineLayout.get(),0u,1u,&kernelDescriptorSet.get(),nullptr);

					// Ker Image First Axis FFT
					driver->bindComputePipeline(firstKernelFFTPipeline.get());
					FFTClass::dispatchHelper(driver,kernelPipelineLayout.get(),fftPushConstants[0],fftDispatchInfo[0]);

					// Ker Image Last Axis FFT
					driver->bindComputePipeline(lastKernelFFTPipeline.get());
					FFTClass::dispatchHelper(driver,kernelPipelineLayout.get(),fftPushConstants[1],fftDispatchInfo[1]);

					// normalization and shuffle
					driver->bindComputePipeline(kernelNormalizationPipeline.get());
					{
						NormalizationPushConstants normalizationPC;
						normalizationPC.stride = fftPushConstants[1].output_strides;
						normalizationPC.bitreverse_shift[0] = 32-core::findMSB(paddedKernelExtent.width);
						normalizationPC.bitreverse_shift[1] = 32-core::findMSB(paddedKernelExtent.height);
						normalizationPC.bloomIntensity = param.bloomIntensity;
						driver->pushConstants(kernelNormalizationPipeline->getLayout(),ICPUSpecializedShader::ESS_COMPUTE,0u,sizeof(normalizationPC),&normalizationPC);
						const uint32_t dispatchSizeX = (paddedKernelExtent.width-1u)/16u+1u;
						const uint32_t dispatchSizeY = (paddedKernelExtent.height-1u)/16u+1u;
						driver->dispatch(dispatchSizeX,dispatchSizeY,colorChannelsFFT);
					}
					FFTClass::defaultBarrier();
				}
			}

			uint32_t outImageByteOffset[EII_COUNT];
			// bind shader resources
			{
				// create descriptor set
				auto descriptorSet = driver->createDescriptorSet(core::smart_refctd_ptr(sharedDescriptorSetLayout));
				// write descriptor set
				{
					IGPUDescriptorSet::SDescriptorInfo infos[SharedDescriptorSetDescCount+EII_COUNT*2u-2u+colorChannelsFFT];
					auto attachBufferImageRange = [param,&infos](auto* pInfo, IGPUBuffer* buff, uint64_t offset, uint64_t pixelByteSize) -> void
					{
						pInfo->desc = core::smart_refctd_ptr<IGPUBuffer>(buff);
						pInfo->buffer = {offset,param.width*param.height*pixelByteSize};
					};
					auto attachWholeBuffer = [&infos](auto* pInfo, IGPUBuffer* buff) -> void
					{
						pInfo->desc = core::smart_refctd_ptr<IGPUBuffer>(buff);
						pInfo->buffer = {0ull,buff->getMemoryReqs().vulkanReqs.size};
					};
					IGPUDescriptorSet::SWriteDescriptorSet writes[SharedDescriptorSetDescCount] =
					{
						{descriptorSet.get(),0u,0u,denoiserInputCount,EDT_STORAGE_BUFFER,infos+0},
						{descriptorSet.get(),1u,0u,denoiserInputCount,EDT_STORAGE_BUFFER,infos+EII_COUNT},
						{descriptorSet.get(),2u,0u,1u,EDT_STORAGE_BUFFER,infos+EII_COUNT*2u},
						{descriptorSet.get(),3u,0u,1u,EDT_STORAGE_BUFFER,infos+EII_COUNT*2u+1u},
						{descriptorSet.get(),4u,0u,colorChannelsFFT,EDT_COMBINED_IMAGE_SAMPLER,infos+EII_COUNT*2u+2u}
					};
					uint64_t interleavedPixelBytesize = getTexelOrBlockBytesize<EF_R16G16B16A16_SFLOAT>();
					attachBufferImageRange(writes[0].info+EII_COLOR,colorPixelBuffer.getObject(),inImageByteOffset[EII_COLOR],interleavedPixelBytesize);
					if (denoiserInputCount>EII_ALBEDO)
						attachBufferImageRange(writes[0].info+EII_ALBEDO,albedoPixelBuffer.getObject(),inImageByteOffset[EII_ALBEDO],interleavedPixelBytesize);
					if (denoiserInputCount>EII_NORMAL)
						attachBufferImageRange(writes[0].info+EII_NORMAL,normalPixelBuffer.getObject(),inImageByteOffset[EII_NORMAL],interleavedPixelBytesize);
					for (uint32_t j=0u; j<denoiserInputCount; j++)
					{
						outImageByteOffset[j] = j*param.width*param.height*forcedOptiXFormatPixelStride;
						attachBufferImageRange(writes[1].info+j,temporaryPixelBuffer.getObject(),outImageByteOffset[j],forcedOptiXFormatPixelStride);
						if (j==0u)
							infos[EII_COUNT].buffer.size = fftScratchSize;
					}
					attachWholeBuffer(writes[2].info,histogramBuffer.get());
					attachWholeBuffer(writes[3].info,intensityBuffer.getObject());
					for (auto j=0u; j<colorChannelsFFT; j++)
					{
						writes[4].info[j].desc = core::smart_refctd_ptr(kernelNormalizedSpectrums[j]);
						//writes[0].info[4].image.imageLayout = ;
						writes[4].info[j].image.sampler = nullptr; //immutable
					}
					driver->updateDescriptorSets(SharedDescriptorSetDescCount,writes,0u,nullptr);
				}
				// bind descriptor set (for all shaders)
				driver->bindDescriptorSets(video::EPBP_COMPUTE,sharedPipelineLayout.get(),0u,1u,&descriptorSet.get(),nullptr);
			}
			// upload the constants to the GPU
			driver->pushConstants(sharedPipelineLayout.get(), video::IGPUSpecializedShader::ESS_COMPUTE, 0u, sizeof(CommonPushConstants), &shaderConstants);
			// compute shader pre-preprocess (transform normals and compute luminosity)
			{
				// bind deinterleave pipeline
				driver->bindComputePipeline(deinterleavePipeline.get());
				// dispatch
				const uint32_t workgroupCounts[2] = {(param.width+kComputeWGSize-1u)/kComputeWGSize,param.height};
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
				const auto buffersUsed = calcDenoiserBuffersNeeded(param.denoiserType);
				if (check_error(!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::acquireAndGetPointers(bufferLinks,bufferLinks+buffersUsed,m_cudaStream)),"Error when mapping OpenGL Buffers to CUdeviceptr!"))
					return error_code;

				auto unmapBuffers = [&m_cudaStream,buffersUsed,&bufferLinks]() -> void
				{
					void* scratch[calcDenoiserBuffersNeeded(EII_NORMAL)*sizeof(CUgraphicsResource)];
					if (check_error(!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::releaseResourcesToGraphics(scratch,bufferLinks,bufferLinks+buffersUsed,m_cudaStream)), "Error when unmapping CUdeviceptr back to OpenGL!"))
						exit(error_code);
				};
				core::SRAIIBasedExiter<decltype(unmapBuffers)> exitRoutine(unmapBuffers);

				// set up denoiser
				auto& denoiser = denoisers[param.denoiserType];
				if (denoiser.m_denoiser->setup(m_cudaStream, denoiseTileDimsWithOverlap, denoiserState, denoiser.stateSize, scratch, denoiser.scratchSize, denoiser.stateOffset) != OPTIX_SUCCESS)
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
					denoiserInputs[k].data = temporaryPixelBuffer.asBuffer.pointer+outImageByteOffset[k];
					denoiserInputs[k].width = param.width;
					denoiserInputs[k].height = param.height;
					denoiserInputs[k].rowStrideInBytes = param.width * forcedOptiXFormatPixelStride;
					denoiserInputs[k].format = forcedOptiXFormat;
					denoiserInputs[k].pixelStrideInBytes = forcedOptiXFormatPixelStride;

				}

				denoiserOutput.data = colorPixelBuffer.asBuffer.pointer+inImageByteOffset[EII_COLOR];
				denoiserOutput.width = param.width;
				denoiserOutput.height = param.height;
				denoiserOutput.rowStrideInBytes = param.width * forcedOptiXFormatPixelStride;
				denoiserOutput.format = forcedOptiXFormat;
				denoiserOutput.pixelStrideInBytes = forcedOptiXFormatPixelStride;
#if 1 // for easy debug with renderdoc disable optix stuff
				//invoke
				if (denoiser.m_denoiser->tileAndInvoke(
					m_cudaStream,
					&denoiserParams,
					denoiserInputs,
					denoiserInputCount,
					&denoiserOutput,
					scratch,
					denoiser.scratchSize,
					overlap,
					tileWidth,
					tileHeight
				) != OPTIX_SUCCESS)
				{
					os::Printer::log(makeImageIDString(i) + "Could not invoke the denoiser sucessfully, skipping image!", ELL_ERROR);
					continue;
				}
#else
				driver->copyBuffer(temporaryPixelBuffer.getObject(),colorPixelBuffer.getObject(),inImageByteOffset[EII_COLOR],outImageByteOffset[EII_COLOR],denoiserInputs[EII_COLOR].rowStrideInBytes*param.height);
#endif
			}

			// compute post-processing
			{
				// let the shaders know we're in the second phase now
				shaderConstants.flags &= ~0b01u;
				driver->pushConstants(sharedPipelineLayout.get(), video::IGPUSpecializedShader::ESS_COMPUTE, offsetof(CommonPushConstants,flags), sizeof(uint32_t), &shaderConstants.flags);
				// Bloom
				uint32_t workgroupCounts[2] = { (param.width+kComputeWGSize-1u)/kComputeWGSize,param.height };
				{
					driver->bindComputePipeline(secondLumaMeterAndFirstFFTPipeline.get());
					// dispatch
					driver->dispatch(param.fftDispatchInfo[0].workGroupCount[0],param.fftDispatchInfo[0].workGroupCount[1],1u);
					COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

					// Y-axis FFT, multiply the spectra together, y-axis iFFT
					driver->bindComputePipeline(convolvePipeline.get());
					{
						const auto& kernelImgExtent = kernelNormalizedSpectrums[0]->getCreationParameters().image->getCreationParameters().extent;
						vec2 kernel_half_pixel_size{0.5f,0.5f};
						kernel_half_pixel_size.x /= kernelImgExtent.width;
						kernel_half_pixel_size.y /= kernelImgExtent.height;
						driver->pushConstants(convolvePipeline->getLayout(),ISpecializedShader::ESS_COMPUTE,offsetof(CommonPushConstants,kernel_half_pixel_size),sizeof(CommonPushConstants::kernel_half_pixel_size),&kernel_half_pixel_size);
					}
					// dispatch
					driver->dispatch(param.fftDispatchInfo[1].workGroupCount[0],param.fftDispatchInfo[1].workGroupCount[1],1u);
					COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

					// bind intensity pipeline
					driver->bindComputePipeline(intensityPipeline.get());
					// dispatch
					driver->dispatch(1u,1u,1u);
					COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
				}
				// Tonemap and interleave the output
				{
					driver->bindComputePipeline(interleaveAndLastFFTPipeline.get());
					driver->dispatch(param.fftDispatchInfo[2].workGroupCount[0],param.fftDispatchInfo[2].workGroupCount[1],1u);
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
			const uint32_t colorBufferBytesize = param.height*param.width*param.colorTexelSize;
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

						driver->copyBuffer(colorPixelBuffer.getObject(),downloadStagingArea->getBuffer(),0u,address,colorBufferBytesize);
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
				using CONVERSION_FILTER = CConvertFormatImageFilter<EF_UNKNOWN,EF_UNKNOWN,asset::CPrecomputedDither,void,true>;

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

					if (!convertFilter.execute(core::execution::par_unseq,&state))
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