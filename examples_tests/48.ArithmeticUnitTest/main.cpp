#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include "../../source/Irrlicht/COpenGLDriver.h"


using namespace irr;
using namespace core;
using namespace video;
using namespace asset;



template<typename T>
struct and
{
	using type_t = T;
	_IRR_STATIC_INLINE_CONSTEXPR T IdentityElement = ~0ull; // this should be a reinterpret cast

	inline T operator()(T left, T right) { return left & right; }

	_IRR_STATIC_INLINE_CONSTEXPR const char* name = "and";
};
template<typename T>
struct xor
{
	using type_t = T;
	_IRR_STATIC_INLINE_CONSTEXPR T IdentityElement = 0ull; // this should be a reinterpret cast

	inline T operator()(T left, T right) { return left ^ right; }

	_IRR_STATIC_INLINE_CONSTEXPR const char* name = "xor";
};
template<typename T>
struct or
{
	using type_t = T;
	_IRR_STATIC_INLINE_CONSTEXPR T IdentityElement = 0ull; // this should be a reinterpret cast

	inline T operator()(T left, T right) { return left | right; }

	_IRR_STATIC_INLINE_CONSTEXPR const char* name = "or";
};
template<typename T>
struct add
{
	using type_t = T;
	_IRR_STATIC_INLINE_CONSTEXPR T IdentityElement = T(0);

	inline T operator()(T left, T right) { return left + right; }

	_IRR_STATIC_INLINE_CONSTEXPR const char* name = "add";
};
template<typename T>
struct mul
{
	using type_t = T;
	_IRR_STATIC_INLINE_CONSTEXPR T IdentityElement = T(1);

	inline T operator()(T left, T right) { return left * right; }

	_IRR_STATIC_INLINE_CONSTEXPR const char* name = "mul";
};
template<typename T>
struct min
{
	using type_t = T;
	_IRR_STATIC_INLINE_CONSTEXPR T IdentityElement = std::numeric_limits<T>::max();

	inline T operator()(T left, T right) { return std::min<T>(left, right); }

	_IRR_STATIC_INLINE_CONSTEXPR const char* name = "min";
};
template<typename T>
struct max
{
	using type_t = T;
	_IRR_STATIC_INLINE_CONSTEXPR T IdentityElement = std::numeric_limits<T>::lowest();

	inline T operator()(T left, T right) { return std::max<T>(left, right); }

	_IRR_STATIC_INLINE_CONSTEXPR const char* name = "max";
};


//subgroup method emulations on the CPU, to verify the results of the GPU methods
template<class OP>
struct emulatedSubgroupReduction
{
	using type_t = typename OP::type_t;

	inline type_t operator()(const type_t* workgroupData, const uint32_t localInvocationIndex, uint32_t subgroupSize)
	{
		auto subgroupData = workgroupData+(localInvocationIndex&(-subgroupSize));
		type_t retval = subgroupData[0];
		for (auto i=1u; i<subgroupSize; i++)
			retval = OP()(retval,subgroupData[i]);
		return retval;
	}

	_IRR_STATIC_INLINE_CONSTEXPR const char* name = "subgroup reduction";
};
template<class OP>
struct emulatedSubgroupScanExclusive
{
	using type_t = typename OP::type_t;

	inline type_t operator()(const type_t* workgroupData, const uint32_t localInvocationIndex, uint32_t subgroupSize)
	{
		auto subgroupData = workgroupData+(localInvocationIndex&(-subgroupSize));
		auto subgroupInvocationID = localInvocationIndex&(subgroupSize-1u);
		type_t retval = OP::IdentityElement;
		for (auto i=0u; i<subgroupInvocationID; i++)
			retval = OP()(retval, subgroupData[i]);
		return retval;
	}

	_IRR_STATIC_INLINE_CONSTEXPR const char* name = "subgroup exclusive scan";
};
template<class OP>
struct emulatedSubgroupScanInclusive
{
	using type_t = typename OP::type_t;

	inline type_t operator()(const type_t* workgroupData, const uint32_t localInvocationIndex, uint32_t subgroupSize)
	{
		auto subgroupData = workgroupData+(localInvocationIndex&(-subgroupSize));
		auto subgroupInvocationID = localInvocationIndex&(subgroupSize-1u);
		type_t retval = OP::IdentityElement;
		for (auto i=0u; i<=subgroupInvocationID; i++)
			retval = OP()(retval, subgroupData[i]);
		return retval;
	}

	_IRR_STATIC_INLINE_CONSTEXPR const char* name = "subgroup inclusive scan";
};


#include "common.glsl"
constexpr uint32_t kBufferSize = BUFFER_DWORD_COUNT*sizeof(uint32_t);

//returns true if result matches
template<template<class> class Arithmetic, template<class> class OP>
bool validateResults_impl(video::IVideoDriver* driver, const uint32_t* inputData, const uint32_t workgroupSize, const uint32_t workgroupCount, video::IGPUBuffer* bufferToDownload)
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
		for (uint32_t workgroupID=0u; success&&workgroupID<workgroupCount; workgroupID++)
		for (uint32_t localInvocationIndex=0u; localInvocationIndex<workgroupSize; localInvocationIndex++)
		{
			constexpr uint32_t subgroupSize = 4u;

			const auto workgroupOffset = workgroupID*workgroupSize;
			uint32_t val = Arithmetic<OP<uint32_t>>()(inputData+workgroupOffset, localInvocationIndex, subgroupSize);
			const auto invocationOffset = workgroupOffset+localInvocationIndex;
			if (val!=dataFromBuffer[invocationOffset])
			{
				os::Printer::log("Failed test #" + std::to_string(workgroupSize) + " (" + Arithmetic<OP<uint32_t>>::name + ")  (" + OP<uint32_t>::name + ")", ELL_ERROR);
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
template<template<class> class Arithmetic>
bool validateResults(video::IVideoDriver* driver, const uint32_t* inputData, const uint32_t workgroupSize, const uint32_t workgroupCount, core::smart_refctd_ptr<IGPUBuffer>* const buffers)
{
	bool passed = validateResults_impl<Arithmetic,and>(driver, inputData, workgroupSize, workgroupCount, buffers[0].get());
	passed = validateResults_impl<Arithmetic,xor>(driver, inputData, workgroupSize, workgroupCount, buffers[1].get())&&passed;
	passed = validateResults_impl<Arithmetic,or>(driver, inputData, workgroupSize, workgroupCount, buffers[2].get())&&passed;
	passed = validateResults_impl<Arithmetic,add>(driver, inputData, workgroupSize, workgroupCount, buffers[3].get())&&passed;
	passed = validateResults_impl<Arithmetic,mul>(driver, inputData, workgroupSize, workgroupCount, buffers[4].get())&&passed;
	passed = validateResults_impl<Arithmetic,::min>(driver, inputData, workgroupSize, workgroupCount, buffers[5].get())&&passed;
	passed = validateResults_impl<Arithmetic,::max>(driver, inputData, workgroupSize, workgroupCount, buffers[6].get())&&passed;
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

	uint32_t* inputData = new uint32_t[BUFFER_DWORD_COUNT];
	{
		std::mt19937 randGenerator(std::time(0));
		for (uint32_t i=0u; i<BUFFER_DWORD_COUNT; i++)
		{
			// TODO: use random numbers, but right now I need to see whats going on in order to debug
			inputData[i] = i;// randGenerator();
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
		{1u,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
		{2u,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
		{3u,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
		{4u,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
		{5u,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
		{6u,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
		{7u,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
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
	
	///uint32_t totalFailCount = 0;
	///constexpr uint32_t totalTestCount = 1022 * 3 * 7;

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
			bool passed = false;
			switch (i)
			{
				case 0:
					passed = validateResults<emulatedSubgroupReduction>(driver, inputData, workgroupSize, workgroupCount, buffers);
					break;
				case 1:
					passed = validateResults<emulatedSubgroupScanExclusive>(driver, inputData, workgroupSize, workgroupCount, buffers);
					break;
				case 2:
					passed = validateResults<emulatedSubgroupScanInclusive>(driver, inputData, workgroupSize, workgroupCount, buffers);
					break;
			}
			if (passed)
				os::Printer::log("Passed test #" + std::to_string(workgroupSize), ELL_INFORMATION);
			else
				os::Printer::log("Failed test #" + std::to_string(workgroupSize), ELL_INFORMATION);
		}
		driver->endScene();
	}
	os::Printer::log("Result:", ELL_INFORMATION);
	//os::Printer::log("Failed:" + totalFailCount, ELL_INFORMATION);
	//os::Printer::log("Total tests:" + totalTestCount, ELL_INFORMATION);

	delete [] inputData;
	return 0;
}
