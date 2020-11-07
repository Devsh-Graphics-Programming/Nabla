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

template<typename T> T and(T left, T right) { return left & right; }
template<typename T> T xor(T left, T right) { return left ^ right; }
template<typename T> T or(T left, T right) { return left | right; }
template<typename T> T add(T left, T right) { return left + right; }
template<typename T> T mul(T left, T right) { return left * right; }
template<typename T> T min(T left, T right) { return std::min(left, right); }
template<typename T> T max(T left, T right) { return std::max(left, right); }

typedef uint32_t(*arithmeticFuncPtr)(uint32_t, uint32_t);
arithmeticFuncPtr arrayofFunctions[] = {
 and<uint32_t>,
 xor<uint32_t>,
 or <uint32_t>,
 add<uint32_t>,
 mul<uint32_t>,
 min<uint32_t>,
 max<uint32_t> 
};

//subgroup method emulations on the CPU, to verify the results of the GPU methods
template<typename T>
T emulatedSubgroupReduction(const T* workgroupData, const uint32_t localInvocationIndex, uint32_t subgroupSize, arithmeticFuncPtr pFunc)
{
	auto subgroupData = workgroupData+(localInvocationIndex&(-subgroupSize));
	T retval = subgroupData[0];
	for (auto i=1u; i<subgroupSize; i++)
		retval = pFunc(retval,subgroupData[i]);
	return retval;
}
template<typename T>
T emulatedScanExclusive(const T* workgroupData, const uint32_t localInvocationIndex, uint32_t subgroupSize, arithmeticFuncPtr pFunc)
{
	auto subgroupData = workgroupData+(localInvocationIndex&(-subgroupSize));
	auto subgroupInvocationID = localInvocationIndex&(subgroupSize-1u);
	T retval = 0u;
	for (auto i=0u; i<subgroupInvocationID; i++)
		retval = pFunc(retval, subgroupData[i]);
	return retval;
}
template<typename T>
T emulatedScanInclusive(const T* workgroupData, const uint32_t localInvocationIndex, uint32_t subgroupSize, arithmeticFuncPtr pFunc)
{
	auto subgroupData = workgroupData+(localInvocationIndex&(-subgroupSize));
	auto subgroupInvocationID = localInvocationIndex&(subgroupSize-1u);
	T retval = 0u;
	for (auto i=0u; i<=subgroupInvocationID; i++)
		retval = pFunc(retval, subgroupData[i]);
	return retval;
}


#include "common.glsl"
constexpr uint32_t kBufferSize = BUFFER_DWORD_COUNT*sizeof(uint32_t);

//returns true if result matches
template<typename EmulatedFunc,typename arithmeticFunc>
bool validateResults(video::IVideoDriver* driver, const uint32_t* inputData, const uint32_t workgroupSize, const uint32_t workgroupCount, video::IGPUBuffer* bufferToDownload, EmulatedFunc emulatedFunc,  arithmeticFunc pFunc)
{
	constexpr uint64_t timeoutInNanoSeconds = 15000000000u;
	const uint32_t alignment = sizeof(uint32_t);
	auto downloadStagingArea = driver->getDefaultDownStreamingBuffer();
	auto downBuffer = downloadStagingArea->getBuffer();


	bool success = false;


	uint32_t address = std::remove_pointer<decltype(downloadStagingArea)>::type::invalid_address;
	auto unallocatedSize = downloadStagingArea->multi_alloc(1u, &address, &kBufferSize, &alignment);
	if (unallocatedSize)
	{
		os::Printer::log("Could not download the buffer from the GPU!", ELL_ERROR);
		return false;
	}
	driver->copyBuffer(bufferToDownload, downBuffer, 0, address, kBufferSize);

	auto downloadFence = driver->placeFence(true);
	auto result = downloadFence->waitCPU(timeoutInNanoSeconds, true);
	if (result != video::E_DRIVER_FENCE_RETVAL::EDFR_TIMEOUT_EXPIRED && result != video::E_DRIVER_FENCE_RETVAL::EDFR_FAIL)
	{
		success = true;

		if (downloadStagingArea->needsManualFlushOrInvalidate())
			driver->invalidateMappedMemoryRanges({ {downloadStagingArea->getBuffer()->getBoundMemory(),address,kBufferSize} });

		auto dataFromBuffer = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(downloadStagingArea->getBufferPointer())+address);

		// now check if the data obtained has valid values
		for (uint32_t workgroupID=0u; workgroupID<workgroupCount; workgroupID++)
		for (uint32_t localInvocationIndex=0u; localInvocationIndex<workgroupSize; localInvocationIndex++)
		{
			constexpr uint32_t subgroupSize = 4u;

			const auto workgroupOffset = workgroupID*workgroupSize;
			uint32_t val = emulatedFunc(inputData+workgroupOffset, localInvocationIndex, subgroupSize, pFunc);
			const auto invocationOffset = workgroupOffset+localInvocationIndex;
			if (val!=dataFromBuffer[invocationOffset])
			{
				success = false;
				break;
			}
		}
	}
	else
		os::Printer::log("Could not download the buffer from the GPU, fence not signalled!", ELL_ERROR);

	downloadStagingArea->multi_free(1u, &address, &kBufferSize, nullptr);
	return success;

}

int main()
{
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24;
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	params.WindowSize = dimension2d<uint32_t>(1280, 720);
	params.Fullscreen = false;
	params.Vsync = true; //! If supported by target platform
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	params.StreamingDownloadBufferSize = kBufferSize;
	auto device = createDeviceEx(params);

	if (!device)
		return 1; // could not create selected driver.

	video::IVideoDriver* driver = device->getVideoDriver();
	io::IFileSystem* filesystem = device->getFileSystem();
	asset::IAssetManager* am = device->getAssetManager();

	uint32_t* inputData = new uint32_t[BUFFER_DWORD_COUNT];
	{
		std::mt19937 randGenerator(std::time(0));
		for (uint32_t i=0u; i<BUFFER_DWORD_COUNT; i++)
		{
			inputData[i] = randGenerator();
		}
	}
	auto gpuinputDataBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(kBufferSize, inputData);
	
	//create 7 buffers.
	core::smart_refctd_ptr<IGPUBuffer> buffers[7];
	for (size_t i = 0; i < 7; i++)
	{
		buffers[i] = driver->createDeviceLocalGPUBufferOnDedMem(kBufferSize);
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
	constexpr uint32_t pushconstantSize = 64u;
	SPushConstantRange pcRange[1] = { IGPUSpecializedShader::ESS_COMPUTE,0u,pushconstantSize };
	auto pipelineLayout = driver->createGPUPipelineLayout(pcRange, pcRange + pushconstantSize, core::smart_refctd_ptr(gpuDSLayout));

	auto descriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr(gpuDSLayout));
	{
		IGPUDescriptorSet::SDescriptorInfo infos[8];
		infos[0].desc = gpuinputDataBuffer;
		infos[0].buffer = { 0u,kBufferSize };
		for (uint32_t i=1u; i<=7u; i++)
		{
			infos[i].desc = buffers[i - 1];
			infos[i].buffer = { 0u,kBufferSize };

		}
		IGPUDescriptorSet::SWriteDescriptorSet writes[8];
		for (uint32_t i=0u; i<8u; i++)
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
	
	uint32_t totalFailCount = 0;
	constexpr uint32_t totalTestCount = 1022 * 3 * 7;

	//As of now, subgroup size is hardcoded to 4
	//workgroup size is required to be greater or equal to subgroup_size/2
	//max workgroup size is hardcoded to 256
	for (uint32_t workgroupSize=2u; workgroupSize<1024u; workgroupSize++)
	{
		core::smart_refctd_ptr<IGPUComputePipeline> pipelines[3];
		for (uint32_t i=0u; i<3u; i++)
			pipelines[i] = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(pipelineLayout), std::move(getGPUShader(shaderGLSL[i], workgroupSize)));


		driver->beginScene(true);
		for (size_t i = 0; i < 3; i++)
		{
			driver->bindComputePipeline(pipelines[i].get());
			const video::IGPUDescriptorSet* ds = descriptorSet.get();
			driver->bindDescriptorSets(video::EPBP_COMPUTE, pipelines[i]->getLayout(), 0u, 1u, &ds, nullptr);
			uint32_t workgroupCount = BUFFER_DWORD_COUNT/workgroupSize;
			driver->dispatch(workgroupCount, 1, 1);
			video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
			//check results 
			bool passedAllOperations = true;
			for (size_t op_index = 0; op_index < 7; op_index++)
			{
				bool passed = false;
				switch (i)
				{
				case 0:
					passed = validateResults(driver, inputData, workgroupSize, workgroupCount, buffers[i].get(), emulatedSubgroupReduction<uint32_t>, arrayofFunctions[op_index]);
					break;
				case 1:
					passed = validateResults(driver, inputData, workgroupSize, workgroupCount, buffers[i].get(), emulatedScanExclusive<uint32_t>, arrayofFunctions[op_index]);
					break;
				case 2:
					passed = validateResults(driver, inputData, workgroupSize, workgroupCount, buffers[i].get(), emulatedScanInclusive<uint32_t>, arrayofFunctions[op_index]);
					break;
				}
				if (!passed)
				{
					os::Printer::log("Failed test #"+ std::to_string( workgroupSize)+ " (scan type "+std::to_string(i)+")  ("+std::to_string(op_index)+"/7)", ELL_ERROR);
					totalFailCount++;
					passedAllOperations = false;
				}
			}
			if (passedAllOperations)
			{
				os::Printer::log("Passed test #" + std::to_string(workgroupSize), ELL_INFORMATION);
			}
		}
		driver->endScene();
	}
	os::Printer::log("Result:", ELL_INFORMATION);
	os::Printer::log("Failed:" + totalFailCount, ELL_INFORMATION);
	os::Printer::log("Total tests:" + totalTestCount, ELL_INFORMATION);

	delete [] inputData;
	return 0;
}
