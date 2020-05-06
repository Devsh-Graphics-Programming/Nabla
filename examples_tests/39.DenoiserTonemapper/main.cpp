#define _IRR_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <irrlicht.h>

#include "CommandLineHandler.hpp"

#include "../../ext/OptiX/Manager.h"

#include "CommonPushConstants.h"

using namespace irr;
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
	core::smart_refctd_ptr<asset::ICPUImage> image[EII_COUNT] = { nullptr,nullptr,nullptr };
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
/*
	if (check_error(,"!"))
		return error_code;
*/

int main(int argc, char* argv[])
{
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24;
	params.ZBufferBits = 24;
	params.DriverType = video::EDT_OPENGL;
	params.WindowSize = core::dimension2d<uint32_t>(1280, 720);
	params.Fullscreen = false;
	params.Vsync = true;
	params.Doublebuffer = true;
	params.Stencilbuffer = false;
	params.StreamingDownloadBufferSize = 256*1024*1024; // change in Vulkan fo
	auto device = createDeviceEx(params);

	if (check_error(!device,"Could not create Irrlicht Device!"))
		return error_code;

	auto driver = device->getVideoDriver();
	auto smgr = device->getSceneManager();
	auto am = device->getAssetManager();

	auto getArgvFetchedList = [&]()
	{
		core::vector<std::string> arguments;
		arguments.reserve(PROPER_CMD_ARGUMENTS_AMOUNT);
		arguments.emplace_back(argv[0]);
		if (argc>1)
		for (auto i = 1ul; i < argc; ++i)
			arguments.emplace_back(argv[i]);
		else // use default for example
		{
			arguments.emplace_back("-batch");
			arguments.emplace_back("../exampleInputArguments.txt");
		}

		return arguments;
	};

	auto cmdHandler = CommandLineHandler(getArgvFetchedList(), am);

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
	constexpr auto forcedOptiXFormatPixelStride = 6u;
	DenoiserToUse denoisers[EII_COUNT];
	{
		OptixDenoiserOptions opts = { OPTIX_DENOISER_INPUT_RGB,forcedOptiXFormat };
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

	constexpr uint32_t kComputeWGSize = 1024u;
	core::smart_refctd_ptr<IGPUDescriptorSetLayout> sharedDescriptorSetLayout;
	core::smart_refctd_ptr<IGPUPipelineLayout> sharedPipelineLayout;
	core::smart_refctd_ptr<IGPUComputePipeline> deinterleavePipeline,interleavePipeline;
	{
		auto deinterleaveShader = driver->createGPUShader(core::make_smart_refctd_ptr<ICPUShader>(R"===(
#version 450 core
#extension GL_EXT_shader_16bit_storage : require

#include "../ShaderCommon.glsl"

layout(binding = 0, std430) restrict readonly buffer InputBuffer
{
	f16vec4 inBuffer[];
};
layout(binding = 1, std430) restrict writeonly buffer OutputBuffer
{
	float16_t outBuffer[];
};


void main()
{
	vec4 data = vec4(inBuffer[pc.data.inImageTexelOffset[gl_GlobalInvocationID.z]+gl_GlobalInvocationID.y*pc.data.inImageTexelPitch[gl_GlobalInvocationID.z]+gl_GlobalInvocationID.x]);
	if (gl_GlobalInvocationID.z==EII_ALBEDO && all(greaterThan(vec3(0.00000001),data.rgb)))
		data.rgb = vec3(1.0);
	else if (gl_GlobalInvocationID.z==EII_NORMAL)
		data.xyz = normalize(pc.data.normalMatrix*data.xyz);
	repackBuffer[gl_LocalInvocationIndex*SHARED_CHANNELS+0u] = data[0u];
	repackBuffer[gl_LocalInvocationIndex*SHARED_CHANNELS+1u] = data[1u];
	repackBuffer[gl_LocalInvocationIndex*SHARED_CHANNELS+2u] = data[2u];
	barrier();
	memoryBarrierShared();

	const uint outImagePitch = pc.data.imageWidth*SHARED_CHANNELS;
	uint rowOffset = pc.data.outImageOffset[gl_GlobalInvocationID.z]+gl_GlobalInvocationID.y*outImagePitch;

	uint lineOffset = gl_WorkGroupID.x*kComputeWGSize*SHARED_CHANNELS+gl_LocalInvocationIndex;
	if (lineOffset<outImagePitch)
		outBuffer[rowOffset+lineOffset] = float16_t(repackBuffer[gl_LocalInvocationIndex+kComputeWGSize*0u]);
	lineOffset += kComputeWGSize;
	if (lineOffset<outImagePitch)
		outBuffer[rowOffset+lineOffset] = float16_t(repackBuffer[gl_LocalInvocationIndex+kComputeWGSize*1u]);
	lineOffset += kComputeWGSize;
	if (lineOffset<outImagePitch)
		outBuffer[rowOffset+lineOffset] = float16_t(repackBuffer[gl_LocalInvocationIndex+kComputeWGSize*2u]);
}
		)==="));
		auto interleaveShader = driver->createGPUShader(core::make_smart_refctd_ptr<ICPUShader>(R"===(
#version 450 core
#extension GL_EXT_shader_16bit_storage : require

#include "../ShaderCommon.glsl"

layout(binding = 0, std430) restrict readonly buffer InputBuffer
{
	float16_t inBuffer[];
};
layout(binding = 1, std430) restrict writeonly buffer OutputBuffer
{
	f16vec4 outBuffer[];
};


void main()
{
	uint wgOffset = pc.data.outImageOffset[EII_COLOR]+(gl_GlobalInvocationID.y*pc.data.imageWidth+gl_WorkGroupID.x*kComputeWGSize)*SHARED_CHANNELS;

	uint localOffset = gl_LocalInvocationIndex;
	repackBuffer[localOffset] = float(inBuffer[wgOffset+localOffset]);
	localOffset += kComputeWGSize;
	repackBuffer[localOffset] = float(inBuffer[wgOffset+localOffset]);
	localOffset += kComputeWGSize;
	repackBuffer[localOffset] = float(inBuffer[wgOffset+localOffset]);
	barrier();
	memoryBarrierShared();

	bool alive = gl_GlobalInvocationID.x<pc.data.imageWidth;
	vec3 color = vec3(repackBuffer[gl_LocalInvocationIndex*SHARED_CHANNELS+0u],repackBuffer[gl_LocalInvocationIndex*SHARED_CHANNELS+1u],repackBuffer[gl_LocalInvocationIndex*SHARED_CHANNELS+2u]);
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
															{2u,offsetof(SpecializationConstants,enumEII_NORMAL),sizeof(SpecializationConstants::enumEII_NORMAL)}
														}
													),
													core::smart_refctd_ptr(specConstantBuffer),"main",ISpecializedShader::ESS_COMPUTE
												};
		auto deinterleaveSpecializedShader = driver->createGPUSpecializedShader(deinterleaveShader.get(),specInfo);
		auto interleaveSpecializedShader = driver->createGPUSpecializedShader(interleaveShader.get(),specInfo);
		
		IGPUDescriptorSetLayout::SBinding binding[2] = { {0u,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},{1u,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr} };
		sharedDescriptorSetLayout = driver->createGPUDescriptorSetLayout(binding,binding+sizeof(binding)/sizeof(IGPUDescriptorSetLayout::SBinding));
		SPushConstantRange pcRange[1] = {IGPUSpecializedShader::ESS_COMPUTE,0u,sizeof(CommonPushConstants)};
		sharedPipelineLayout = driver->createGPUPipelineLayout(pcRange,pcRange+sizeof(pcRange)/sizeof(SPushConstantRange),core::smart_refctd_ptr(sharedDescriptorSetLayout));

		deinterleavePipeline = driver->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(sharedPipelineLayout),std::move(deinterleaveSpecializedShader));
		interleavePipeline = driver->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(sharedPipelineLayout),std::move(interleaveSpecializedShader));
	}

	const auto inputFilesAmount = cmdHandler.getInputFilesAmount();
	const auto& fileNamesBundle = cmdHandler.getFileNamesBundle();
	const auto& channelNamesBundle = cmdHandler.getChannelNamesBundle();
	const auto& cameraTransformBundle = cmdHandler.getCameraTransformBundle();
	const auto& exposureBiasBundle = cmdHandler.getExposureBiasBundle();
	const auto& denoiserBlendFactorBundle = cmdHandler.getDenoiserBlendFactorBundle();
	const auto& bloomSizeBundle = cmdHandler.getBloomSizeBundle();
	const auto& tonemapperBundle = cmdHandler.getTonemapperBundle();
	const auto& outputFileBundle = cmdHandler.getOutputFileBundle();

	auto makeImageIDString = [&fileNamesBundle](uint32_t i)
	{
		std::string imageIDString("Image Input #");
		imageIDString += std::to_string(i);
		imageIDString += " called \"";
		imageIDString += fileNamesBundle[i];
		imageIDString += "\": ";
		return imageIDString;
	};


	core::vector<ImageToDenoise> images(inputFilesAmount);
	// load images
	uint32_t maxResolution[EII_COUNT][2] = { 0 };
	{
		asset::IAssetLoader::SAssetLoadParams lp(0ull,nullptr,IAssetLoader::ECF_DUPLICATE_REFERENCES);
		for (size_t i=0; i<inputFilesAmount; i++)
		{
			auto image_bundle = am->getAsset(std::string("../../media/OpenEXR/")+fileNamesBundle[i], lp);
			if (image_bundle.isEmpty())
			{
				auto imageIDString = makeImageIDString(i);
				os::Printer::log(imageIDString+"Could not load from file!", ELL_ERROR);
				continue;
			}

			auto& outParam = images[i];

			auto contents = image_bundle.getContents();
			for (auto it=contents.first; it!=contents.second; it++)
			{
				auto ass = *it;
				assert(ass);
				auto metadata = ass->getMetadata();
				if (strcmp(metadata->getLoaderName(),COpenEXRImageMetadata::LoaderName)!=0)
					continue;

				auto exrmeta = static_cast<COpenEXRImageMetadata*>(metadata);
				auto beginIt = channelNamesBundle[i].begin();
				auto inputIx = std::distance(beginIt,std::find(beginIt,channelNamesBundle[i].end(),exrmeta->getName()));
				if (inputIx>=channelNamesBundle[i].size())
					continue;

				outParam.image[inputIx] = core::smart_refctd_ptr_static_cast<ICPUImage>(ass);
			}
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
					uint32_t bytesize = extent.height*region.bufferRowLength*outParam.colorTexelSize;
					if (bytesize>params.StreamingDownloadBufferSize)
					{
						os::Printer::log(imageIDString + "Image too large to download from GPU in one piece!", ELL_ERROR);
						outParam = {};
						continue;
					}
				}

				outParam.denoiserType = EII_COLOR;

				outParam.width = extent.width;
				outParam.height = extent.height;
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

			maxResolution[outParam.denoiserType][0] = core::max(maxResolution[outParam.denoiserType][0],outParam.width);
			maxResolution[outParam.denoiserType][1] = core::max(maxResolution[outParam.denoiserType][1],outParam.height);
		}
	}

#define DENOISER_BUFFER_COUNT 4u //had to change to a define cause lambda was complaining about it not being a constant expression when capturing
	// keep all CUDA links in an array (less code to map/unmap
	cuda::CCUDAHandler::GraphicsAPIObjLink<video::IGPUBuffer> bufferLinks[DENOISER_BUFFER_COUNT];
	// set-up denoisers
	constexpr size_t intensityValuesSize = sizeof(float);
	auto& intensityBuffer = bufferLinks[0];
	auto& denoiserState = bufferLinks[0];
	auto& scratchBuffer = bufferLinks[1];
	auto& temporaryPixelBuffer = bufferLinks[2];
	auto& imagePixelBuffer = bufferLinks[3];
	size_t denoiserStateBufferSize = 0ull;
	{
		size_t scratchBufferSize = 0ull;
		size_t pixelBufferSize = 0ull;
		for (uint32_t i=0u; i<EII_COUNT; i++)
		{
			auto& denoiser = denoisers[i].m_denoiser;
			if (maxResolution[i][0]==0u || maxResolution[i][1]==0u)
			{
				denoiser = nullptr;
				continue;
			}

			OptixDenoiserSizes m_denoiserMemReqs;
			if (denoiser->computeMemoryResources(&m_denoiserMemReqs, maxResolution[i])!=OPTIX_SUCCESS)
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
			scratchBufferSize = core::max(scratchBufferSize,denoisers[i].scratchSize = m_denoiserMemReqs.recommendedScratchSizeInBytes);
			pixelBufferSize = core::max(pixelBufferSize,core::max(asset::getTexelOrBlockBytesize(EF_R32G32B32A32_SFLOAT),(i+1u)*forcedOptiXFormatPixelStride)*maxResolution[i][0]*maxResolution[i][1]);
		}
		std::string message = "Total VRAM consumption for Denoiser algorithm: ";
		os::Printer::log(message+std::to_string(denoiserStateBufferSize+scratchBufferSize+pixelBufferSize), ELL_INFORMATION);

		if (check_error(pixelBufferSize==0ull,"No input files at all!"))
			return error_code;

		denoiserState = driver->createDeviceLocalGPUBufferOnDedMem(denoiserStateBufferSize+intensityValuesSize);
		if (check_error(!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::registerBuffer(&denoiserState)),"Could not register buffer for Denoiser states!"))
			return error_code;
		scratchBuffer = driver->createDeviceLocalGPUBufferOnDedMem(scratchBufferSize);
		if (check_error(!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::registerBuffer(&scratchBuffer)),"Could not register buffer for Denoiser scratch memory!"))
			return error_code;
		temporaryPixelBuffer = driver->createDeviceLocalGPUBufferOnDedMem(pixelBufferSize);
		if (check_error(!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::registerBuffer(&temporaryPixelBuffer)),"Could not register buffer for Denoiser scratch memory!"))
			return error_code;
	}

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
			shaderConstants.normalMatrix = cameraTransformBundle[i];
		}

		// upload image channels and register their buffer
		{
			asset::ICPUBuffer* buffersToUpload[EII_COUNT];
			for (uint32_t j=0u; j<denoiserInputCount; j++)
				buffersToUpload[j] = param.image[j]->getBuffer();
			auto gpubuffers = driver->getGPUObjectsFromAssets(buffersToUpload,buffersToUpload+denoiserInputCount);

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
		{
			// bind shader resources
			{
				// create descriptor set
				auto descriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr(sharedDescriptorSetLayout));
				// write descriptor set
				{
					IGPUDescriptorSet::SDescriptorInfo infos[2];
					infos[0].desc = core::smart_refctd_ptr<IGPUBuffer>(imagePixelBuffer.getObject());
					infos[0].buffer = {0u,imagePixelBuffer.getObject()->getMemoryReqs().vulkanReqs.size};
					infos[1].desc = core::smart_refctd_ptr<IGPUBuffer>(temporaryPixelBuffer.getObject());
					infos[1].buffer = {0u,temporaryPixelBuffer.getObject()->getMemoryReqs().vulkanReqs.size};
					IGPUDescriptorSet::SWriteDescriptorSet writes[2] = { {descriptorSet.get(),0u,0u,1u,EDT_STORAGE_BUFFER,infos+0},{descriptorSet.get(),1u,0u,1u,EDT_STORAGE_BUFFER,infos+1} };
					driver->updateDescriptorSets(sizeof(writes)/sizeof(IGPUDescriptorSet::SWriteDescriptorSet),writes,0u,nullptr);
				}
				// bind descriptor set (for all shaders)
				driver->bindDescriptorSets(video::EPBP_COMPUTE,sharedPipelineLayout.get(),0u,1u,&descriptorSet.get(),nullptr);
			}
			// compute shader pre-preprocess (transform normals and TODO: compute luminosity)
			{
				// bind compute pipeline
				driver->bindComputePipeline(deinterleavePipeline.get());
				// dispatch
				driver->dispatch((param.width+kComputeWGSize-1u)/kComputeWGSize,param.height,denoiserInputCount);
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

				// set up optix image
				OptixImage2D denoiserInputs[EII_COUNT];
				for (uint32_t j=0u; j<denoiserInputCount; j++)
				{				
					denoiserInputs[j].data = temporaryPixelBuffer.asBuffer.pointer+shaderConstants.outImageOffset[j]*sizeof(uint16_t); // sizeof(float16_t)
					denoiserInputs[j].width = param.width;
					denoiserInputs[j].height = param.height;
					denoiserInputs[j].rowStrideInBytes = param.width*forcedOptiXFormatPixelStride;
					denoiserInputs[j].pixelStrideInBytes = forcedOptiXFormatPixelStride; // either 0 or the value that corresponds to a dense packing of format = NO CHOICE
					denoiserInputs[j].format = forcedOptiXFormat;
				}
				//
				{
					// set up denoiser
					auto& denoiser = denoisers[param.denoiserType];
					if (denoiser.m_denoiser->setup(m_cudaStream,&param.width,denoiserState,denoiser.stateSize,scratchBuffer,denoiser.scratchSize,denoiser.stateOffset)!=OPTIX_SUCCESS)
					{
						os::Printer::log(makeImageIDString(i) + "Could not setup the denoiser for the image resolution and denoiser buffers, skipping image!", ELL_ERROR);
						continue;
					}
					// compute intensity (TODO: tonemapper after image upload)
					const auto intensityBufferOffset = denoiserStateBufferSize;
					if (denoiser.m_denoiser->computeIntensity(m_cudaStream,denoiserInputs+EII_COLOR,intensityBuffer,scratchBuffer,denoiser.scratchSize,intensityBufferOffset)!=OPTIX_SUCCESS)
					{
						os::Printer::log(makeImageIDString(i) + "Could not setup the denoiser for the image resolution and denoiser buffers, skipping image!", ELL_ERROR);
						continue;
					}
					// invoke
					{
						OptixDenoiserParams denoiserParams = {};
						denoiserParams.blendFactor = denoiserBlendFactorBundle[i];
						denoiserParams.denoiseAlpha = 0u;
						denoiserParams.hdrIntensity = intensityBuffer.asBuffer.pointer+intensityBufferOffset;
						OptixImage2D denoiserOutput;
						denoiserOutput.data = imagePixelBuffer.asBuffer.pointer+shaderConstants.inImageTexelOffset[EII_COLOR];
						denoiserOutput.width = param.width;
						denoiserOutput.height = param.height;
						denoiserOutput.rowStrideInBytes = param.width*forcedOptiXFormatPixelStride;
						denoiserOutput.pixelStrideInBytes = forcedOptiXFormatPixelStride; // either 0 or the value that corresponds to a dense packing of format = NO CHOICE
						denoiserOutput.format = forcedOptiXFormat;
						if (denoiser.m_denoiser->invoke(m_cudaStream,&denoiserParams,denoiserInputs,denoiserInputs+denoiserInputCount,&denoiserOutput,scratchBuffer,denoiser.scratchSize)!=OPTIX_SUCCESS)
						{
							os::Printer::log(makeImageIDString(i) + "Could not invoke the denoiser sucessfully, skipping image!", ELL_ERROR);
							continue;
						}
					}
				}

				// unmap buffer (implicit from the SRAIIExiter destructor)
			}

			// compute post-processing
			{
				// TODO: Bloom (FoV vs. Constant)
				{
				}
				// TODO: Tonemap
				{
				}
				driver->bindComputePipeline(interleavePipeline.get());
				driver->dispatch((param.width+kComputeWGSize-1u)/kComputeWGSize,param.height,1u);
				// issue a full memory barrier (or at least all buffer read/write barrier)
				COpenGLExtensionHandler::extGlMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
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

					// download buffer
					{
						const uint32_t alignment = 4096u; // common page size
						auto unallocatedSize = downloadStagingArea->multi_alloc(std::chrono::nanoseconds(timeoutInNanoSeconds), 1u, &address, &colorBufferBytesize, &alignment);
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

			// save image
			IAssetWriter::SAssetWriteParams wp(imageView.get());
			am->writeAsset(outputFileBundle[i].c_str(), wp);

			// destroy link to CPUBuffer's data (we need to free it)
			imageView->convertToDummyObject(~0u);

			// free the staging area allocation (no fence, we've already waited on it)
			downloadStagingArea->multi_free(1u,&address,&colorBufferBytesize,nullptr);

			// destroy image (implicit)
		}
	}

	return 0;
}