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
	core::vector<cuda::CCUDAHandler::GraphicsAPIObjLink<video::IGPUBuffer> > bufferLinks;
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
		core::vector<asset::ICPUBuffer*> buffersToUpload;
		buffersToUpload.reserve(images.size() * EII_COUNT);
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

				outParam.denoiserType = EII_COLOR;
				buffersToUpload.emplace_back(colorImage->getBuffer());

				auto extent = colorImage->getCreationParameters().extent;
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
				{
					outParam.denoiserType = EII_ALBEDO;
					buffersToUpload.emplace_back(albedoImage->getBuffer());
				}
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
				{
					outParam.denoiserType = EII_NORMAL;
					buffersToUpload.emplace_back(normalImage->getBuffer());
				}
			}

			maxResolution[outParam.denoiserType][0] = core::max(maxResolution[outParam.denoiserType][0],outParam.width);
			maxResolution[outParam.denoiserType][1] = core::max(maxResolution[outParam.denoiserType][1],outParam.height);
		}
		// upload image data buffers to GPU
		const auto* _begin = buffersToUpload.data();
		auto gpubuffers = driver->getGPUObjectsFromAssets(_begin,_begin+buffersToUpload.size());
		for (size_t i=0; i<inputFilesAmount; i++)
		{
			auto& outParam = images[i];
			for (uint32_t j=0u; j<EII_COUNT; j++)
			{
				auto img = outParam.image[j];
				if (!img)
					continue;

				auto offsetPair = gpubuffers->operator[](j);

				auto buffer = core::smart_refctd_ptr<video::IGPUBuffer>(offsetPair->getBuffer());
				auto found = std::find_if(bufferLinks.begin(),bufferLinks.end(),[&buffer](const auto& l){return l.getObject()==buffer.get();});
				if (found==bufferLinks.end())
				{
					cuda::CCUDAHandler::GraphicsAPIObjLink<video::IGPUBuffer> link = core::smart_refctd_ptr(buffer);
					if (check_error(!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::registerBuffer(&link,CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY)),"Could not register buffers containing image data with CUDA!"))
						return error_code;
					bufferLinks.push_back(std::move(link));
				}

				outParam.imgData[j].offset = offsetPair->getOffset();
				outParam.imgData[j].buffer = std::move(buffer);
			}
		}
		// make sure stuff doesn't stay around in cache
		for (auto it=gpubuffers->begin(); it!=gpubuffers->end(); it++,_begin++)
			am->removeCachedGPUObject(*_begin,*it);
	}


	// set-up denoisers
	cuda::CCUDAHandler::GraphicsAPIObjLink<video::IGPUBuffer> denoiserState,denoiserScratch;
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

		// upload image channels
		// TODO: transform image normals
		// register buffer
		// map buffer
		// set up optix image
		{
			// set up denoiser
			// compute intensity (TODO: tonemapper after image upload)
			// invoke
		}
		// unmap buffer
		// TODO: Bloom (FoV vs. Constant)
		// download buffer
		// create image
		// save image
		// destroy
	}

	return 0;
}