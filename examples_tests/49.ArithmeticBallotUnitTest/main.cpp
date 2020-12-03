#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include "../../source/Irrlicht/COpenGLDriver.h"


using namespace irr;
using namespace core;
using namespace video;
using namespace asset;

//workgroup methods - since there are no subgroup methods for bit count

uint32_t bitCount(uint32_t input)
{
	return input & 1;
}

struct emulatedWorkgroupReduction
{
	inline void operator()(uint32_t* outputData, uint32_t workgroupSize, uint32_t subgroupSize)
	{
		uint32_t bitC = 0;
		/*for (auto i=1u; i<workgroupSize; i++)
			bitC += bitCount<uint32_t>(1);*/
		std::fill(outputData,outputData+workgroupSize,bitC);
	}

	_IRR_STATIC_INLINE_CONSTEXPR const char* name = "workgroup reduction";
};
struct emulatedWorkgroupScanExclusive
{
	inline void operator()(uint32_t* outputData, uint32_t workgroupSize, uint32_t subgroupSize)
	{
		uint32_t bitC = 0;
		//outputData[0u] = OP::IdentityElement;
		//for (auto i=1u; i<workgroupSize; i++)
		//	outputData[i] = OP()(outputData[i-1u],workgroupData[i-1u]);
		//uint32_t bitC = 0;
		//for (auto i = 1u; i < workgroupSize; i++) 
		//	bitC += bitCount<uint32_t>(1);
		std::fill(outputData, outputData + workgroupSize, bitC);


	}

	_IRR_STATIC_INLINE_CONSTEXPR const char* name = "workgroup exclusive scan";
};
struct emulatedWorkgroupScanInclusive
{

	inline void operator()(uint32_t* outputData, uint32_t workgroupSize, uint32_t subgroupSize)
	{
		uint32_t bitC = 0;
	/*	outputData[0u] = workgroupData[0u];
		for (auto i=1u; i<workgroupSize; i++)
			outputData[i] = OP()(outputData[i-1u],workgroupData[i]);*/
		std::fill(outputData, outputData + workgroupSize, bitC);

	}

	_IRR_STATIC_INLINE_CONSTEXPR const char* name = "workgroup inclusive scan";
};


#include "common.glsl"
constexpr uint32_t kBufferSize = BUFFER_DWORD_COUNT*sizeof(uint32_t);

//returns true if result matches
template<class Arithmetic>
bool validateResults(video::IVideoDriver* driver, const uint32_t workgroupSize, const uint32_t workgroupCount, video::IGPUBuffer* bufferToDownload)
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
		constexpr uint32_t subgroupSize = 4u;
		uint32_t* tmp = new uint32_t[workgroupSize];
		for (uint32_t workgroupID=0u; success&&workgroupID<workgroupCount; workgroupID++)
		{
			const auto workgroupOffset = workgroupID*workgroupSize;
			Arithmetic()(tmp,workgroupSize,subgroupSize);
			for (uint32_t localInvocationIndex=0u; localInvocationIndex<workgroupSize; localInvocationIndex++)
			if (tmp[localInvocationIndex]!=dataFromBuffer[workgroupOffset+localInvocationIndex])
			{
				auto correctRes = dataFromBuffer[workgroupOffset + localInvocationIndex];
				os::Printer::log("Failed test #" + std::to_string(workgroupSize) + " (" + Arithmetic::name + ")", ELL_ERROR);
				success = false;
				//break;
			}
		}
		delete[] tmp;
	}
	else
		os::Printer::log("Could not download the buffer from the GPU, fence not signalled!", ELL_ERROR);

	downloadStagingArea->multi_free(1u, &address, &kBufferSize, nullptr);
	return success;

}
template<class Arithmetic>
bool runTest(video::IVideoDriver* driver, video::IGPUComputePipeline* pipeline, const video::IGPUDescriptorSet* ds, const uint32_t workgroupSize, core::smart_refctd_ptr<IGPUBuffer> buffer)
{
	if (pipeline == nullptr) assert(false);	//com
	driver->bindComputePipeline(pipeline);
	driver->bindDescriptorSets(video::EPBP_COMPUTE,pipeline->getLayout(),0u,1u,&ds,nullptr);
	const uint32_t workgroupCount = BUFFER_DWORD_COUNT/workgroupSize;
	driver->dispatch(workgroupCount, 1, 1);
	video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT|GL_SHADER_STORAGE_BARRIER_BIT);
	//check results 
	bool passed = validateResults<Arithmetic>(driver, workgroupSize, workgroupCount, buffer.get());
	return passed;
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

	//buffer with results from the gpu
	core::smart_refctd_ptr<IGPUBuffer> buffer=  driver->createDeviceLocalGPUBufferOnDedMem(kBufferSize);
	

	IGPUDescriptorSetLayout::SBinding binding = { 0u,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr };
	auto gpuDSLayout = driver->createGPUDescriptorSetLayout(&binding, &binding + 1);
	constexpr uint32_t pushconstantSize = 12;
	SPushConstantRange pcRange[1] = { IGPUSpecializedShader::ESS_COMPUTE,0u,pushconstantSize };
	auto pipelineLayout = driver->createGPUPipelineLayout(pcRange, pcRange + pushconstantSize, core::smart_refctd_ptr(gpuDSLayout));

	auto descriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr(gpuDSLayout));
	{
		IGPUDescriptorSet::SDescriptorInfo info;
		info.desc = buffer;
		info.buffer = { 0u,kBufferSize };
		
		IGPUDescriptorSet::SWriteDescriptorSet write = { descriptorSet.get(),0u,0u,1u,EDT_STORAGE_BUFFER, &info };
		
		driver->updateDescriptorSets(1, &write, 0u, nullptr);
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
	GLSLCodeWithWorkgroup shaderGLSL[] =
	{
		getShaderGLSL("../testWorkgroupReduce.comp"),
		getShaderGLSL("../testWorkgroupExclusive.comp"),
		getShaderGLSL("../testWorkgroupInclusive.comp")
	};
	constexpr auto kTestTypeCount = sizeof(shaderGLSL)/sizeof(GLSLCodeWithWorkgroup);


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

	//max workgroup size is hardcoded to 1024
	uint32_t totalFailCount = 0;
	const auto ds = descriptorSet.get();
	for (uint32_t workgroupSize=1u; workgroupSize<=1024u; workgroupSize++)
	{
		core::smart_refctd_ptr<IGPUComputePipeline> pipelines[kTestTypeCount];
		for (uint32_t i=0u; i<kTestTypeCount; i++)
			pipelines[i] = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(pipelineLayout), std::move(getGPUShader(shaderGLSL[i], workgroupSize)));

		bool passed = true;

		driver->beginScene(true);
		const video::IGPUDescriptorSet* ds = descriptorSet.get();
		passed = runTest<emulatedWorkgroupReduction>(driver,pipelines[0u].get(),descriptorSet.get(),workgroupSize,buffer)&&passed;
		passed = runTest<emulatedWorkgroupScanExclusive>(driver,pipelines[1u].get(),descriptorSet.get(),workgroupSize, buffer)&&passed;
		passed = runTest<emulatedWorkgroupScanInclusive>(driver,pipelines[2u].get(),descriptorSet.get(),workgroupSize, buffer)&&passed;

		if (passed)
			os::Printer::log("Passed test #" + std::to_string(workgroupSize), ELL_INFORMATION);
		else
		{
			totalFailCount++;
			os::Printer::log("Failed test #" + std::to_string(workgroupSize), ELL_INFORMATION);
		}
		driver->endScene();
	}
	os::Printer::log("==========Result==========", ELL_INFORMATION);
	os::Printer::log("Fail Count: " + std::to_string(totalFailCount), ELL_INFORMATION);

	return 0;
}
