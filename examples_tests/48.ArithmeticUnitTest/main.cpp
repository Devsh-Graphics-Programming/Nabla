#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include "../../source/Irrlicht/COpenGLDriver.h"


using namespace irr;
using namespace core;
using namespace video;
using namespace asset;

enum TestOperation {
	TO_AND = 1,
	TO_XOR,
	TO_OR,
	TO_ADD,
	TO_MUL,
	TO_MIN,
	TO_MAX
};

template<typename T, typename OP, uint32_t subgroupSize>
T emulatedSubgroupReduction(const T* data, const uint32_t globalInvocationIndex)
{
	auto subgroupID = globalInvocationIndex / subgroupSize;
	//auto subgroupOffset = data + subgroupID * subgroupSize;
	T retval = data[0];
	for (auto i = 1u; i < subgroupSize; i++)
		retval = OP(retval, data[i]);
	return retval;
}
uint32_t add(uint32_t a, uint32_t b)
{
	return a + b;
}

constexpr uint32_t BUFFER_SIZE = 128u * 1024u * 1024u;
//returns true if result matches
template<typename Func>
bool validateResults(video::IVideoDriver* driver, const uint32_t* inputData, video::IGPUBuffer* bufferToDownload)
{
	constexpr uint64_t timeoutInNanoSeconds = 15000000000u;
	const uint32_t alignment = sizeof(uint32_t);
	auto downloadStagingArea = driver->getDefaultDownStreamingBuffer();
	auto downBuffer = downloadStagingArea->getBuffer();



	uint32_t address = std::remove_pointer<decltype(downloadStagingArea)>::type::invalid_address;
	auto unallocatedSize = downloadStagingArea->multi_alloc(1u, &address, &BUFFER_SIZE, &alignment);
	if (unallocatedSize)
	{
		os::Printer::log("Could not download the buffer from the GPU!", ELL_ERROR);
		return false;
	}
	driver->copyBuffer(bufferToDownload, downBuffer, 0, address, BUFFER_SIZE);
	auto downloadFence = driver->placeFence(true);
	auto result = downloadFence->waitCPU(timeoutInNanoSeconds, true);
	if (result == video::E_DRIVER_FENCE_RETVAL::EDFR_TIMEOUT_EXPIRED || result == video::E_DRIVER_FENCE_RETVAL::EDFR_FAIL)
	{
		os::Printer::log("Could not download the buffer from the GPU, fence not signalled!", ELL_ERROR);
		downloadStagingArea->multi_free(1u, &address, &BUFFER_SIZE, nullptr);
		return false;
	}
	if (downloadStagingArea->needsManualFlushOrInvalidate())
		driver->invalidateMappedMemoryRanges({ {downloadStagingArea->getBuffer()->getBoundMemory(),address,BUFFER_SIZE} });

	auto dataFromBuffer = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(downloadStagingArea->getBufferPointer()) + address);
	//now check if the data obtained has valid values
	uint32_t* end = dataFromBuffer + BUFFER_SIZE / sizeof(uint32_t);
	uint32_t invocationIndex = 0;
	for (uint32_t* i = dataFromBuffer; i < end; i++)
	{
		uint32_t val = *(inputData + invocationIndex);	//TODO instead the template function and compare result
		/*if (val != *i)
			goto validationEndFalse;*/
		invocationIndex++;
	}
	downloadStagingArea->multi_free(1u, &address, &BUFFER_SIZE, nullptr);
	return true;

validationEndFalse:
	downloadStagingArea->multi_free(1u, &address, &BUFFER_SIZE, nullptr);
	return false;

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
	params.StreamingDownloadBufferSize = BUFFER_SIZE * 2;
	auto device = createDeviceEx(params);

	if (!device)
		return 1; // could not create selected driver.

	video::IVideoDriver* driver = device->getVideoDriver();
	io::IFileSystem* filesystem = device->getFileSystem();
	asset::IAssetManager* am = device->getAssetManager();

	uint32_t* inputData = new uint32_t[BUFFER_SIZE / sizeof(uint32_t)];

	auto inputDataBuffer = make_smart_refctd_ptr<ICPUBuffer>(BUFFER_SIZE);
	{
		uint32_t* ptr = static_cast<uint32_t*>(inputDataBuffer->getPointer());
		for (size_t i = 0; i < BUFFER_SIZE / sizeof(uint32_t); i++)
		{
			auto memAddr = ptr + i;
			uint32_t randomValue = std::rand();
			*memAddr = randomValue;
			inputData[i] = randomValue;
		}
	}
	core::smart_refctd_ptr<IGPUBuffer> gpuinputDataBuffer = core::smart_refctd_ptr<IGPUBuffer>(driver->getGPUObjectsFromAssets(&inputDataBuffer, &inputDataBuffer + 1)->front()->getBuffer());
	//create 7 buffers.
	core::smart_refctd_ptr<IGPUBuffer> buffers[7];
	for (size_t i = 0; i < 7; i++)
	{
		buffers[i] = driver->createDeviceLocalGPUBufferOnDedMem(BUFFER_SIZE);
	}

	IGPUDescriptorSetLayout::SBinding binding[8] = {
		{0u,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},	//input with randomized numbers
		{TO_AND,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
		{TO_XOR,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
		{TO_OR,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
		{TO_ADD,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
		{TO_MUL,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
		{TO_MIN,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
		{TO_MAX,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
	};
	auto gpuDSLayout = driver->createGPUDescriptorSetLayout(binding, binding + 8);
	constexpr uint32_t pushconstantSize = 1;
	SPushConstantRange pcRange[1] = { IGPUSpecializedShader::ESS_COMPUTE,0u,pushconstantSize };
	auto pipelineLayout = driver->createGPUPipelineLayout(pcRange, pcRange + pushconstantSize, core::smart_refctd_ptr(gpuDSLayout));

	auto descriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr(gpuDSLayout));
	{
		IGPUDescriptorSet::SDescriptorInfo infos[8];
		infos[0].desc = gpuinputDataBuffer;
		infos[0].buffer = { 0u, BUFFER_SIZE };
		for (size_t i = 1; i <= 7; i++)
		{
			infos[i].desc = buffers[i - 1];
			infos[i].buffer = { 0u,BUFFER_SIZE };

		}
		IGPUDescriptorSet::SWriteDescriptorSet writes[8];
		for (uint32_t i = 0u; i < 8; i++)
			writes[i] = { descriptorSet.get(),i,0u,1u,EDT_STORAGE_BUFFER,infos + i };
		driver->updateDescriptorSets(8, writes, 0u, nullptr);
	}
	struct GLSLCodeWithWorkgroup {
		uint32_t workgroup_definition_position;
		std::string glsl;
	};
	constexpr const char* symbolsToReplace = "????";
	auto getShaderGLSL = [&](const char* filePath)
	{
		std::ifstream file(filePath);
		std::stringstream buff; buff << file.rdbuf();
		std::string shaderCode = buff.str();
		uint32_t wgPos = shaderCode.find(symbolsToReplace, 0);
		GLSLCodeWithWorkgroup ret = { wgPos,shaderCode };
		return ret;
	};
	GLSLCodeWithWorkgroup shaderGLSL[3] =
	{
		getShaderGLSL("../testReduce.comp"),
		getShaderGLSL("../testInclusive.comp"),
		getShaderGLSL("../testExclusive.comp"),
	};


auto getGPUShader = [&](GLSLCodeWithWorkgroup glsl, uint32_t wg_count)
{
	auto alteredGLSL = glsl.glsl.replace(glsl.workgroup_definition_position, 4, std::to_string(wg_count));
	auto shaderUnspecialized = core::make_smart_refctd_ptr<asset::ICPUShader>(alteredGLSL.data());
	asset::ISpecializedShader::SInfo specinfo(nullptr, nullptr, "main", IGPUSpecializedShader::ESS_COMPUTE,"../file.comp");
	auto cs = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(shaderUnspecialized), std::move(specinfo));
	auto cs_rawptr = cs.get();
	core::smart_refctd_ptr<IGPUSpecializedShader> shader = driver->getGPUObjectsFromAssets(&cs_rawptr, &cs_rawptr + 1)->front();
	return shader;
};
	for (size_t current_Workgroup = 2; current_Workgroup < 1024; current_Workgroup++)
	{


		auto reducePipeline = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(pipelineLayout), std::move(getGPUShader(shaderGLSL[0], current_Workgroup)));
		auto inclusivePipeline = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(pipelineLayout), std::move(getGPUShader(shaderGLSL[0], current_Workgroup)));
		auto exclusicePipeline = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(pipelineLayout), std::move(getGPUShader(shaderGLSL[0], current_Workgroup)));


		core::smart_refctd_ptr<IGPUComputePipeline> pipelines[3] = { reducePipeline ,inclusivePipeline ,exclusicePipeline };


		if (device->run())
		{
			driver->beginScene(true);
			for (size_t i = 0; i < 3; i++)	//1 because skipping reduce! UNDO THIS
			{
				driver->bindComputePipeline(pipelines[i].get());
				const video::IGPUDescriptorSet* ds = descriptorSet.get();
				driver->bindDescriptorSets(video::EPBP_COMPUTE, pipelines[i]->getLayout(), 0u, 1u, &ds, nullptr);
				driver->dispatch(BUFFER_SIZE / (sizeof(uint32_t)), 1, 1);
				video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_ALL_BARRIER_BITS);
				//check results 
				for (size_t i = TO_AND; i <= TO_MAX; i++)
				{
					validateResults<	uint32_t/*replace with method*/		>(driver, inputData, buffers[i - 1].get());
				}
			}



			driver->endScene();
		}

	}
	return 0;
}
