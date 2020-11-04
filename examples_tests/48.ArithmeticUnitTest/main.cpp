#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include "../../source/Irrlicht/COpenGLDriver.h"


using namespace irr;
using namespace core;
using namespace video;
using namespace asset;

enum TestOperation {
	TO_AND=1,
	TO_XOR,
	TO_OR,
	TO_ADD,
	TO_MUL,
	TO_MIN,
	TO_MAX
};

constexpr uint32_t BUFFER_SIZE = 128u * 1024u * 1024u;
//returns true if result matches
template<typename Func>
bool ValidateResults(video::IVideoDriver* driver, video::IGPUBuffer* bufferToDownload, Func CPUfunctionCounterpart)
{
	constexpr uint64_t timeoutInNanoSeconds = 15000000000u;
	const uint32_t alignment = sizeof(float);
	auto downloadStagingArea = driver->getDefaultDownStreamingBuffer();
	auto downBuffer = downloadStagingArea->getBuffer();

	
	
		uint32_t address = std::remove_pointer<decltype(downloadStagingArea)>::type::invalid_address;
		auto unallocatedSize = downloadStagingArea->multi_alloc(1u, &address, &BUFFER_SIZE, &alignment);
		if (unallocatedSize)
		{
			os::Printer::log("Could not download the buffer from the GPU!", ELL_ERROR);
			return nullptr;
		}
		driver->copyBuffer(bufferToDownload, downBuffer, 0, address, BUFFER_SIZE);
		auto downloadFence = driver->placeFence(true);
		auto result = downloadFence->waitCPU(timeoutInNanoSeconds, true);
		if (result == video::E_DRIVER_FENCE_RETVAL::EDFR_TIMEOUT_EXPIRED || result == video::E_DRIVER_FENCE_RETVAL::EDFR_FAIL)
		{
			os::Printer::log("Could not download the buffer from the GPU, fence not signalled!", ELL_ERROR);
			downloadStagingArea->multi_free(1u, &address, &BUFFER_SIZE, nullptr);
			return nullptr;
		}
		if (downloadStagingArea->needsManualFlushOrInvalidate())
			driver->invalidateMappedMemoryRanges({ {downloadStagingArea->getBuffer()->getBoundMemory(),address,BUFFER_SIZE} });
		
		auto dataFromBuffer = reinterpret_cast<float*>(downloadStagingArea->getBufferPointer()) + address;
		//now check if the data is obtained
		float* end = dataFromBuffer + BUFFER_SIZE / sizeof(float);
		for (float* i = dataFromBuffer; i < end; i++)
		{
			auto val = CPUfunctionCounterpart();
		}

		downloadStagingArea->multi_free(1u, &address, &BUFFER_SIZE, nullptr);

}
int main()
{
	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check irr::SIrrlichtCreationParameters
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	params.WindowSize = dimension2d<uint32_t>(1280, 720);
	params.Fullscreen = false;
	params.Vsync = true; //! If supported by target platform
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	params.StreamingDownloadBufferSize = BUFFER_SIZE;
	auto device = createDeviceEx(params);

	if (!device)
		return 1; // could not create selected driver.

	video::IVideoDriver* driver = device->getVideoDriver();
	io::IFileSystem* filesystem = device->getFileSystem();
	asset::IAssetManager* am = device->getAssetManager();

	uint32_t* inputData = new uint32_t[BUFFER_SIZE/sizeof(uint32_t)];

	auto inputDataBuffer = make_smart_refctd_ptr<ICPUBuffer>(BUFFER_SIZE);
	{
		uint32_t* ptr = static_cast<uint32_t*>(inputDataBuffer->getPointer());
		for (size_t i = 0; i < BUFFER_SIZE/sizeof(float); i++)
		{
			auto memAddr = ptr + i;
			uint32_t randomValue = std::rand();
			*memAddr = randomValue;
			inputData[i] = static_cast<float>(randomValue);
		}
	}
	core::smart_refctd_ptr<IGPUBuffer> gpuinputDataBuffer =core::smart_refctd_ptr<IGPUBuffer>( driver->getGPUObjectsFromAssets(&inputDataBuffer, &inputDataBuffer + 1)->front()->getBuffer());
	//create 7 buffers.
	core::smart_refctd_ptr<IGPUBuffer> buffers[7];
	for (size_t i = 0; i < 7; i++)
	{
		buffers[i] = driver->createDeviceLocalGPUBufferOnDedMem(BUFFER_SIZE); 
	}

	IGPUDescriptorSetLayout::SBinding binding[15] = {
		{0u,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},	//input with randomized numbers
		{TO_AND,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
		{TO_XOR,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
		{TO_OR,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
		{TO_ADD,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
		{TO_MUL,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
		{TO_MIN,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
		{TO_MAX,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
	};
	auto gpuDSLayout = driver->createGPUDescriptorSetLayout(binding, binding + 15);
	constexpr uint32_t pushconstantSize = 1;
	SPushConstantRange pcRange[1] = { IGPUSpecializedShader::ESS_COMPUTE,0u,pushconstantSize };
	auto pipelineLayout = driver->createGPUPipelineLayout(pcRange, pcRange + sizeof(pcRange) / pushconstantSize, core::smart_refctd_ptr(gpuDSLayout));

	auto descriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr(gpuDSLayout));
	{
		IGPUDescriptorSet::SDescriptorInfo infos[8];
		infos[0].desc = gpuinputDataBuffer;
		infos[0].buffer = { 0u, BUFFER_SIZE };
		for (size_t i = 1; i <= 7; i++)
		{
			infos[i].desc = buffers[i];
			infos[i].buffer = { 0u,BUFFER_SIZE };

		}
		IGPUDescriptorSet::SWriteDescriptorSet writes[8];
		for (uint32_t i = 0u; i < 8; i++)
			writes[i] = { descriptorSet.get(),i,0u,1u,EDT_STORAGE_BUFFER,infos + i };
		driver->updateDescriptorSets(8, writes, 0u, nullptr);
	}
	auto getGPUShader = [&](irr::io::path& filePath)
	{
		auto file = core::smart_refctd_ptr<io::IReadFile>(filesystem->createAndOpenFile(filePath));
		asset::IAssetLoader::SAssetLoadParams lp;
		auto cs_bundle = am->getAsset(filePath.c_str(), lp);
		auto cs = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*cs_bundle.getContents().begin());
		auto cs_rawptr = cs.get();
		core::smart_refctd_ptr<IGPUSpecializedShader> shader = driver->getGPUObjectsFromAssets(&cs_rawptr, &cs_rawptr + 1)->front();
		return shader;
	};


	auto reducePipeline = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(pipelineLayout),		std::move(getGPUShader(io::path("../testReduce.comp"))));
	auto inclusivePipeline = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(pipelineLayout),	std::move(getGPUShader(io::path("../testInclusive.comp"))));
	auto exclusicePipeline = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(pipelineLayout),	std::move(getGPUShader(io::path("../testExclusive.comp"))));


	core::smart_refctd_ptr<IGPUComputePipeline> pipelines[3] = { reducePipeline ,inclusivePipeline ,exclusicePipeline };

	

	if (device->run())
	{
		driver->beginScene(true);
		for (size_t i = 0; i < 3; i++)
		{
			driver->bindComputePipeline(pipelines[i].get());
			const video::IGPUDescriptorSet* ds = descriptorSet.get();
			driver->bindDescriptorSets(video::EPBP_COMPUTE, pipelines[i]->getLayout(), 0u, 1u, &ds, nullptr);
			driver->dispatch(BUFFER_SIZE/(sizeof(float)),1,1);
			video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
			//check results 

		}



		driver->endScene();
	}

	return 0;
}
