#define _IRR_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <irrlicht.h>

#include "CommandLineHandler.hpp"

#include "../../ext/OptiX/Manager.h"

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
	uint32_t colorBufferBytesize = 0u; // includes padding, row strides, etc.
	E_IMAGE_INPUT denoiserType = EII_COUNT;
	core::smart_refctd_ptr<asset::ICPUImage> image[EII_COUNT] = { nullptr,nullptr,nullptr };
	asset::SBufferBinding<IGPUBuffer> imgData[EII_COUNT] = { {0,nullptr},{0,nullptr},{0,nullptr} };
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

	core::smart_refctd_ptr<ext::OptiX::IDenoiser> m_denoiser[EII_COUNT];
	{
		OptixDenoiserOptions opts = { OPTIX_DENOISER_INPUT_RGB,OPTIX_PIXEL_FORMAT_HALF3 };
		m_denoiser[EII_COLOR] = m_optixContext->createDenoiser(&opts);
		if (check_error(!m_denoiser, "Could not create Optix Color Denoiser!"))
			return error_code;
		opts.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO;
		m_denoiser[EII_ALBEDO] = m_optixContext->createDenoiser(&opts);
		if (check_error(!m_denoiser, "Could not create Optix Color-Albedo Denoiser!"))
			return error_code;
		opts.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL;
		m_denoiser[EII_NORMAL] = m_optixContext->createDenoiser(&opts);
		if (check_error(!m_denoiser, "Could not create Optix Color-Albedo-Normal Denoiser!"))
			return error_code;
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
				if (inputIx>=EII_COUNT)
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
					uint32_t bytesize = extent.height*region.bufferRowLength*asset::getTexelOrBlockBytesize(colorCreationParams.format);
					if (bytesize>params.StreamingDownloadBufferSize)
					{
						os::Printer::log(imageIDString + "Image too large to download from GPU in one piece!", ELL_ERROR);
						outParam = {};
						continue;
					}
					outParam.colorBufferBytesize = bytesize;
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

	// keep all CUDA links in an array (less code to map/unmap
	cuda::CCUDAHandler::GraphicsAPIObjLink<video::IGPUBuffer> bufferLinks[EII_COUNT+2u];
	const auto links_begin = bufferLinks;
	// set-up denoisers
	auto& denoiserState = bufferLinks[0];
	auto& denoiserScratch = bufferLinks[1];
	constexpr uint32_t denoiserBufferCount = 2u;
	{
		size_t stateBufferSize = 0ull;
		size_t scratchBufferSize = 0ull;
		for (uint32_t i=0u; i<EII_COUNT; i++)
		{
			if (maxResolution[i][0]==0u || maxResolution[i][1]==0u)
			{
				m_denoiser[i] = nullptr;
				continue;
			}

			OptixDenoiserSizes m_denoiserMemReqs;
			if (m_denoiser[i]->computeMemoryResources(&m_denoiserMemReqs, maxResolution[i])!=OPTIX_SUCCESS)
			{
				static const char* errorMsgs[EII_COUNT] = {	"Failed to compute Color-Denoiser Memory Requirements!",
															"Failed to compute Color-Albedo-Denoiser Memory Requirements!",
															"Failed to compute Color-Albedo-Normal-Denoiser Memory Requirements!"};
				os::Printer::log(errorMsgs[i],ELL_ERROR);
				m_denoiser[i] = nullptr;
				continue;
			}

			stateBufferSize += m_denoiserMemReqs.stateSizeInBytes;
			scratchBufferSize = core::max(scratchBufferSize,m_denoiserMemReqs.recommendedScratchSizeInBytes);
		}
		std::string message = "Total VRAM consumption for Denoiser algorithm: ";
		os::Printer::log(message+std::to_string(stateBufferSize+scratchBufferSize), ELL_INFORMATION);

		if (check_error(stateBufferSize+scratchBufferSize==0ull,"No input files at all!"))
			return error_code;

		denoiserState = driver->createDeviceLocalGPUBufferOnDedMem(stateBufferSize);
		if (check_error(!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::registerBuffer(&denoiserState,CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD)),"Could not register buffer for Denoiser states!"))
			return error_code;
		denoiserScratch = driver->createDeviceLocalGPUBufferOnDedMem(scratchBufferSize);
		if (check_error(!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::registerBuffer(&denoiserScratch,CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD)),"Could not register buffer for Denoiser scratch memory!"))
			return error_code;
	}

	// do the processing
	for (size_t i=0; i<inputFilesAmount; i++)
	{
		auto& param = images[i];
		if (param.denoiserType>=EII_COUNT)
			continue;

		// upload image channels and register their buffers
		uint32_t gpuBufferCount = denoiserBufferCount;
		{
			asset::ICPUBuffer* buffersToUpload[EII_COUNT];
			for (uint32_t j=0u; j<param.denoiserType; j++)
				buffersToUpload[j] = param.image[j]->getBuffer();
			auto gpubuffers = driver->getGPUObjectsFromAssets(buffersToUpload,buffersToUpload+param.denoiserType);

			// register with cuda and deregister from cache
			for (uint32_t j=0u; j<param.denoiserType; j++)
			{
				auto offsetPair = gpubuffers->operator[](j);
				// make sure cache doesn't retain the GPU object paired to CPU object (could have used a custom IGPUObjectFromAssetConverter derived class with overrides to achieve this)
				am->removeCachedGPUObject(buffersToUpload[j],offsetPair);

				auto buffer = core::smart_refctd_ptr<video::IGPUBuffer>(offsetPair->getBuffer());

				const auto links_end = bufferLinks+gpuBufferCount;
				auto found = std::find_if(links_begin,links_end,[&buffer](const auto& l) {return l.getObject() == buffer.get(); });
				if (found==links_end)
				{
					*found = core::smart_refctd_ptr(buffer);
					if (check_error(!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::registerBuffer(found, CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY)), "Could not register buffers containing image data with CUDA!"))
						return error_code;
					gpuBufferCount++;
				}

				param.imgData[j].offset = offsetPair->getOffset();
				param.imgData[j].buffer = std::move(buffer);
			}

		}

		// process
		{
			// create descriptor set
			// write descriptor set
			// TODO: transform image normals
			{
				// bind compute pipeline
				// bind descriptor set with SSBO
				// dispatch
				// issue a full memory barrier (or at least all buffer read barrier)
			}

			// map buffer
			if (check_error(!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::acquireAndGetPointers(bufferLinks,bufferLinks+gpuBufferCount,m_cudaStream)),"Error when mapping OpenGL Buffers to CUdeviceptr!"))
				return error_code;

			// set up optix image
			OptixImage2D denoiserInputs[EII_COUNT];
			for (uint32_t j=0u; j<param.denoiserType; j++)
			{
				const auto& image = param.image[j];
				const auto& imgCreationParams = image->getCreationParameters();
				auto format = imgCreationParams.format;
				// assert a few things to ensure sanity
				{
					auto dims = asset::getBlockDimensions(format);
					assert(dims.x==1 && dims.y==1 && dims.z==1);
				}
				auto regions = image->getRegions();
				
				// find our CUDA link
				const auto links_end = bufferLinks+gpuBufferCount;
				const auto& data = param.imgData[j];
				auto found = std::find_if(links_begin,links_end,[&data](const auto& l) {return l.getObject() == data.buffer.get(); });
				assert(found!=links_end);
				denoiserInputs[j].data = found->asBuffer.pointer+data.offset;
				denoiserInputs[j].width = param.width;
				denoiserInputs[j].height = param.height;
				denoiserInputs[j].rowStrideInBytes = asset::getTexelOrBlockBytesize(format)*regions.begin()[0].bufferRowLength;
				denoiserInputs[j].pixelStrideInBytes = 0; // either 0 or the value that corresponds to a dense packing of format = NO CHOICE
				denoiserInputs[j].format = ext::OptiX::irrFormatToOptiX(format);
			}
			//
			{
				// set up denoiser
				// compute intensity (TODO: tonemapper after image upload)
				// invoke
			}

			// unmap buffer
			void* scratch[(EII_COUNT+denoiserBufferCount)*sizeof(CUgraphicsResource)];
			if (check_error(!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::releaseResourcesToGraphics(scratch,bufferLinks,bufferLinks+gpuBufferCount,m_cudaStream)),"Error when unmapping CUdeviceptr back to OpenGL!"))
				return error_code;

			// TODO: Bloom (FoV vs. Constant)
			{
			}
			// delete descriptor set
		}

		constexpr auto outputFormat = EF_R16G16B16A16_SFLOAT;
		{
			auto downloadStagingArea = driver->getDefaultDownStreamingBuffer();
			uint32_t address = std::remove_pointer<decltype(downloadStagingArea)>::type::invalid_address; // remember without initializing the address to be allocated to invalid_address you won't get an allocation!

			// create image
			ICPUImage::SCreationParams imgParams;
			imgParams.flags = static_cast<ICPUImage::E_CREATE_FLAGS>(0u); // no flags
			imgParams.type = ICPUImage::ET_2D;
			imgParams.format = outputFormat;
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
					auto unallocatedSize = downloadStagingArea->multi_alloc(std::chrono::nanoseconds(timeoutInNanoSeconds), 1u, &address, &param.colorBufferBytesize, &alignment);
					if (unallocatedSize)
					{
						os::Printer::log(makeImageIDString(i)+"Could not download the buffer from the GPU!",ELL_ERROR);
						continue;
					}

					driver->copyBuffer(param.imgData[EII_COLOR].buffer.get(),downloadStagingArea->getBuffer(),param.imgData[EII_COLOR].offset,address,param.colorBufferBytesize);
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
				auto cpubufferalias = core::make_smart_refctd_ptr<asset::CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t> > >(param.colorBufferBytesize, data, core::adopt_memory);
				image->setBufferAndRegions(std::move(cpubufferalias),regions);

				// wait for download fence and then invalidate the CPU cache
				{
					auto result = downloadFence->waitCPU(timeoutInNanoSeconds,true);
					if (result==E_DRIVER_FENCE_RETVAL::EDFR_TIMEOUT_EXPIRED||result==E_DRIVER_FENCE_RETVAL::EDFR_FAIL)
					{
						os::Printer::log(makeImageIDString(i)+"Could not download the buffer from the GPU, fence not signalled!",ELL_ERROR);
						downloadStagingArea->multi_free(1u, &address, &param.colorBufferBytesize, nullptr);
						continue;
					}
					if (downloadStagingArea->needsManualFlushOrInvalidate())
						driver->invalidateMappedMemoryRanges({{downloadStagingArea->getBuffer()->getBoundMemory(),address,param.colorBufferBytesize}});
				}
			}

			// save image
			IAssetWriter::SAssetWriteParams wp(image.get());
			am->writeAsset(outputFileBundle[i].c_str(), wp);

			// destroy link to CPUBuffer's data (we need to free it)
			image->convertToDummyObject(~0u);

			// free the staging area allocation (no fence, we've already waited on it)
			downloadStagingArea->multi_free(1u,&address,&param.colorBufferBytesize,nullptr);

			// destroy image (implicit)
		}
	}

	return 0;
}