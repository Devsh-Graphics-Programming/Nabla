#define _IRR_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"

using namespace nbl;
using namespace core;
using namespace video;
using namespace asset;


template<typename T>
struct and_op
{
	using type_t = T;
	static inline constexpr T IdentityElement = std::bit_cast<T,uint32_t>(~0ull);

	inline T operator()(T left, T right) { return left & right; }
	static inline constexpr bool runOPonFirst = false;
	static inline constexpr const char* name = "and";
};
template<typename T>
struct xor_op
{
	using type_t = T;
	static inline const T IdentityElement = std::bit_cast<T,uint32_t>(0ull);

	inline T operator()(T left, T right) { return left ^ right; }
	static inline constexpr bool runOPonFirst = false;
	static inline constexpr const char* name = "xor";
};
template<typename T>
struct or_op
{
	using type_t = T;
	static inline const T IdentityElement = std::bit_cast<T,uint32_t>(0ull);

	inline T operator()(T left, T right) { return left | right; }
	static inline constexpr bool runOPonFirst = false;
	static inline constexpr const char* name = "or";
};
template<typename T>
struct add_op
{
	using type_t = T;
	static inline constexpr T IdentityElement = T(0);

	inline T operator()(T left, T right) { return left + right; }
	static inline constexpr bool runOPonFirst = false;
	static inline constexpr const char* name = "add";
};
template<typename T>
struct mul_op
{
	using type_t = T;
	static inline constexpr T IdentityElement = T(1);

	inline T operator()(T left, T right) { return left * right; }
	static inline constexpr bool runOPonFirst = false;
	static inline constexpr const char* name = "mul";
};
template<typename T>
struct min_op
{
	using type_t = T;
	static inline constexpr T IdentityElement = std::numeric_limits<T>::max();

	inline T operator()(T left, T right) { return std::min<T>(left, right); }
	static inline constexpr bool runOPonFirst = false;
	static inline constexpr const char* name = "min";
};
template<typename T>
struct max_op
{
	using type_t = T;
	static inline constexpr T IdentityElement = std::numeric_limits<T>::lowest();

	inline T operator()(T left, T right) { return std::max<T>(left, right); }
	static inline constexpr bool runOPonFirst = false;
	static inline constexpr const char* name = "max";
};
template<typename T>
struct ballot : add_op<T> {};


//subgroup method emulations on the CPU, to verify the results of the GPU methods
template<class CRTP, typename T>
struct emulatedSubgroupCommon
{
	using type_t = T;

	inline void operator()(type_t* outputData, const type_t* workgroupData, uint32_t workgroupSize, uint32_t subgroupSize)
	{
		for (uint32_t pseudoSubgroupID=0u; pseudoSubgroupID<workgroupSize; pseudoSubgroupID+=subgroupSize)
		{
			type_t* outSubgroupData = outputData+pseudoSubgroupID;
			const type_t* subgroupData = workgroupData+pseudoSubgroupID;
			CRTP::impl(outSubgroupData,subgroupData,core::min<uint32_t>(subgroupSize,workgroupSize-pseudoSubgroupID));
		}
	}
};
template<class OP>
struct emulatedSubgroupReduction : emulatedSubgroupCommon<emulatedSubgroupReduction<OP>,typename OP::type_t>
{
	using type_t = typename OP::type_t;

	static inline void impl(type_t* outSubgroupData, const type_t* subgroupData, const uint32_t clampedSubgroupSize)
	{
		type_t red = subgroupData[0];
		for (auto i=1u; i<clampedSubgroupSize; i++)
			red = OP()(red,subgroupData[i]);
		std::fill(outSubgroupData,outSubgroupData+clampedSubgroupSize,red);
	}
	static inline constexpr const char* name = "subgroup reduction";
};
template<class OP>
struct emulatedSubgroupScanExclusive : emulatedSubgroupCommon<emulatedSubgroupScanExclusive<OP>,typename OP::type_t>
{
	using type_t = typename OP::type_t;

	static inline void impl(type_t* outSubgroupData, const type_t* subgroupData, const uint32_t clampedSubgroupSize)
	{
		outSubgroupData[0u] = OP::IdentityElement;
		for (auto i=1u; i<clampedSubgroupSize; i++)
			outSubgroupData[i] = OP()(outSubgroupData[i-1u],subgroupData[i-1u]);
	}
	static inline constexpr const char* name = "subgroup exclusive scan";
};
template<class OP>
struct emulatedSubgroupScanInclusive : emulatedSubgroupCommon<emulatedSubgroupScanInclusive<OP>,typename OP::type_t>
{
	using type_t = typename OP::type_t;

	static inline void impl(type_t* outSubgroupData, const type_t* subgroupData, const uint32_t clampedSubgroupSize)
	{
		outSubgroupData[0u] = subgroupData[0u];
		for (auto i=1u; i<clampedSubgroupSize; i++)
			outSubgroupData[i] = OP()(outSubgroupData[i-1u],subgroupData[i]);
	}
	static inline constexpr const char* name = "subgroup inclusive scan";
};

//workgroup methods
template<class OP>
struct emulatedWorkgroupReduction
{
	using type_t = typename OP::type_t;

	inline void operator()(type_t* outputData, const type_t* workgroupData, uint32_t workgroupSize, uint32_t subgroupSize)
	{
		type_t red = OP::runOPonFirst ? OP()(0, workgroupData[0]) : workgroupData[0];
		for (auto i=1u; i<workgroupSize; i++)
			red = OP()(red,workgroupData[i]);
		std::fill(outputData,outputData+workgroupSize,red);
	}
	static inline constexpr const char* name = "workgroup reduction";
};
template<class OP>
struct emulatedWorkgroupScanExclusive
{
	using type_t = typename OP::type_t;

	inline void operator()(type_t* outputData, const type_t* workgroupData, uint32_t workgroupSize, uint32_t subgroupSize)
	{
		outputData[0u] = OP::IdentityElement;
		for (auto i=1u; i<workgroupSize; i++)
			outputData[i] = OP()(outputData[i-1u],workgroupData[i-1u]);
	}
	static inline constexpr const char* name = "workgroup exclusive scan";
};
template<class OP>
struct emulatedWorkgroupScanInclusive
{
	using type_t = typename OP::type_t;

	inline void operator()(type_t* outputData, const type_t* workgroupData, uint32_t workgroupSize, uint32_t subgroupSize)
	{
		outputData[0u] = workgroupData[0u];
		for (auto i=1u; i<workgroupSize; i++)
			outputData[i] = OP()(outputData[i-1u],workgroupData[i]);
	}
	static inline constexpr const char* name = "workgroup inclusive scan";
};


#include "common.glsl"
constexpr uint32_t kBufferSize = (1u+BUFFER_DWORD_COUNT)*sizeof(uint32_t);

//returns true if result matches
template<template<class> class Arithmetic, template<class> class OP>
bool validateResults(ILogicalDevice* device, const uint32_t* inputData, const uint32_t workgroupSize, const uint32_t workgroupCount, video::IGPUBuffer* bufferToRead, system::ILogger* logger)
{
	bool success = true;

	auto mem = bufferToRead->getBoundMemory();
	if (mem->getMappingCaps()&IDriverMemoryAllocation::EMCF_COHERENT)
	{
		IDriverMemoryAllocation::MappedMemoryRange rng = {mem,0u,kBufferSize};
		device->invalidateMappedMemoryRanges(1u,&rng);
	}

	auto dataFromBuffer = reinterpret_cast<uint32_t*>(mem->getMappedPointer());
	const uint32_t subgroupSize = (*dataFromBuffer++);

	// now check if the data obtained has valid values
	uint32_t* tmp = new uint32_t[workgroupSize];
	uint32_t* ballotInput = new uint32_t[workgroupSize];
	for (uint32_t workgroupID=0u; success&&workgroupID<workgroupCount; workgroupID++)
	{
		const auto workgroupOffset = workgroupID*workgroupSize;
		if constexpr (std::is_same_v<OP<uint32_t>,ballot<uint32_t>>)
		{
			for (auto i=0u; i<workgroupSize; i++)
				ballotInput[i] = inputData[i+workgroupOffset]&0x1u;
			Arithmetic<OP<uint32_t>>()(tmp,ballotInput,workgroupSize,subgroupSize);
		}
		else
			Arithmetic<OP<uint32_t>>()(tmp,inputData+workgroupOffset,workgroupSize,subgroupSize);
		for (uint32_t localInvocationIndex=0u; localInvocationIndex<workgroupSize; localInvocationIndex++)
		if (tmp[localInvocationIndex]!=dataFromBuffer[workgroupOffset+localInvocationIndex])
		{
			logger->log(
				"Failed test #%d  (%s)  (%s) Expected %s got %s",system::ILogger::ELL_ERROR,
				workgroupSize,Arithmetic<OP<uint32_t>>::name,OP<uint32_t>::name,
				std::to_string(tmp[localInvocationIndex]),std::to_string(dataFromBuffer[workgroupOffset+localInvocationIndex])
			);
			success = false;
			break;
		}
	}
	delete[] ballotInput;
	delete[] tmp;

	return success;
}

constexpr const auto outputBufferCount = 8u;

template<template<class> class Arithmetic>
bool runTest(
	ILogicalDevice* device, IGPUQueue* queue, IGPUFence* reusableFence, IGPUCommandBuffer* cmdbuf, IGPUComputePipeline* pipeline, const IGPUDescriptorSet* ds,
	const uint32_t* inputData, const uint32_t workgroupSize, core::smart_refctd_ptr<IGPUBuffer>* const buffers, system::ILogger* logger, bool is_workgroup_test = false)
{
	// TODO: overlap dispatches with memory readbacks (requires multiple copies of `buffers`)

	cmdbuf->begin(IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
	cmdbuf->bindComputePipeline(pipeline);
	cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE,pipeline->getLayout(),0u,1u,&ds,nullptr);
	const uint32_t workgroupCount = BUFFER_DWORD_COUNT/workgroupSize;
	cmdbuf->dispatch(workgroupCount,1,1);
	IGPUCommandBuffer::SBufferMemoryBarrier memoryBarrier[outputBufferCount];
	for (auto i=0u; i<outputBufferCount; i++)
	{
		memoryBarrier[i].barrier.srcAccessMask = EAF_SHADER_WRITE_BIT;
		memoryBarrier[i].barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(EAF_SHADER_WRITE_BIT|EAF_HOST_READ_BIT);
		memoryBarrier[i].srcQueueFamilyIndex = cmdbuf->getQueueFamilyIndex();
		memoryBarrier[i].dstQueueFamilyIndex = cmdbuf->getQueueFamilyIndex();
		memoryBarrier[i].buffer = buffers[i];
		memoryBarrier[i].offset = 0u;
		memoryBarrier[i].size = kBufferSize;
	}
	cmdbuf->pipelineBarrier(
		asset::EPSF_COMPUTE_SHADER_BIT, core::bitflag(asset::EPSF_COMPUTE_SHADER_BIT)|asset::EPSF_HOST_BIT,asset::EDF_NONE,
		0u,nullptr,outputBufferCount,memoryBarrier,0u,nullptr
	);
	cmdbuf->end();

	IGPUQueue::SSubmitInfo submit = {};
	submit.commandBufferCount = 1u;
	submit.commandBuffers = &cmdbuf;
	queue->submit(1u,&submit,reusableFence);
	device->blockForFences(1u,&reusableFence);
	device->resetFences(1u,&reusableFence);

	//check results 
	bool passed = validateResults<Arithmetic,and_op>(device, inputData, workgroupSize, workgroupCount, buffers[0].get(),logger);
	passed = validateResults<Arithmetic,xor_op>(device, inputData, workgroupSize, workgroupCount, buffers[1].get(),logger)&&passed;
	passed = validateResults<Arithmetic,or_op>(device, inputData, workgroupSize, workgroupCount, buffers[2].get(),logger)&&passed;
	passed = validateResults<Arithmetic,add_op>(device, inputData, workgroupSize, workgroupCount, buffers[3].get(),logger)&&passed;
	passed = validateResults<Arithmetic,mul_op>(device, inputData, workgroupSize, workgroupCount, buffers[4].get(),logger)&&passed;
	passed = validateResults<Arithmetic,min_op>(device, inputData, workgroupSize, workgroupCount, buffers[5].get(),logger)&&passed;
	passed = validateResults<Arithmetic,max_op>(device, inputData, workgroupSize, workgroupCount, buffers[6].get(),logger)&&passed;
	if(is_workgroup_test)
	{
		passed = validateResults<Arithmetic,ballot>(device, inputData, workgroupSize, workgroupCount, buffers[7].get(),logger) && passed;
	}

	return passed;
}

// APP

class ArithmeticUnitTestApp : public ApplicationBase
{

public:
	struct Nabla : IUserData
	{
		nbl::core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
		nbl::core::smart_refctd_ptr<nbl::ui::IWindow> window;
		nbl::core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
		nbl::core::smart_refctd_ptr<nbl::video::IAPIConnection> gl;
		nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface;
		nbl::core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
		nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
		nbl::video::IPhysicalDevice* gpuPhysicalDevice;
		std::array<nbl::video::IGPUQueue*, CommonAPI::InitOutput<0u>::EQT_COUNT> queues = { nullptr, nullptr, nullptr };
		nbl::video::IGPUQueue* computeQueue = nullptr;
		nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool> commandPool;
		nbl::core::smart_refctd_ptr<nbl::system::ISystem> system;
		nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
		nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
		nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger;
		nbl::core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;

		nbl::core::smart_refctd_ptr<video::IGPUFence> gpuTransferFence;
		nbl::core::smart_refctd_ptr<video::IGPUFence> gpuComputeFence;
		nbl::video::IGPUObjectFromAssetConverter cpu2gpu;

		core::smart_refctd_ptr<ICPUSpecializedShader> shaderGLSL[6u];
		static constexpr uint64_t kTestTypeCount = sizeof(shaderGLSL) / sizeof(const void*);

		uint32_t workgroupSize = 1u;
		core::smart_refctd_ptr<IGPUPipelineLayout> pipelineLayout;
		core::smart_refctd_ptr<IGPUDescriptorSet> descriptorSet;
		uint32_t* inputData;
		core::smart_refctd_ptr<IGPUBuffer> buffers[outputBufferCount];
		core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
		core::smart_refctd_ptr<IGPUFence> fence;

		//max workgroup size is hardcoded to 1024
		uint32_t totalFailCount = 0;

		auto getGPUShader(const ICPUSpecializedShader* shader, uint32_t wg_count)
		{
			auto overridenUnspecialized = IGLSLCompiler::createOverridenCopy(shader->getUnspecialized(), "#define _NBL_GLSL_WORKGROUP_SIZE_ %d\n", wg_count);
			ISpecializedShader::SInfo specInfo = shader->getSpecializationInfo();
			auto cs = core::make_smart_refctd_ptr<ICPUSpecializedShader>(std::move(overridenUnspecialized), std::move(specInfo));
			return cpu2gpu.getGPUObjectsFromAssets(&cs, &cs + 1, cpu2gpuParams)->front();
			// no need to wait on fences because its only a shader create, does not result in the filling of image or buffers
		};

		void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override
		{
			window = std::move(wnd);
		}
	};

	APP_CONSTRUCTOR(ArithmeticUnitTestApp)

	void onAppInitialized_impl(void* data) override
	{
		Nabla* engine = static_cast<Nabla*>(data);

		CommonAPI::InitOutput<0u> initOutput;
		CommonAPI::Init(initOutput, video::EAT_OPENGL, "Subgroup Arithmetic Test");
		engine->system = std::move(initOutput.system);
		engine->gl = std::move(initOutput.apiConnection);
		engine->logger = std::move(initOutput.logger);
		engine->gpuPhysicalDevice = std::move(initOutput.physicalDevice);
		engine->logicalDevice = std::move(initOutput.logicalDevice);
		engine->queues = std::move(initOutput.queues);
		engine->renderpass = std::move(initOutput.renderpass);
		engine->commandPool = std::move(initOutput.commandPool);
		engine->assetManager = std::move(initOutput.assetManager);
		engine->cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
		engine->utilities = std::move(initOutput.utilities);

		{
			engine->cpu2gpuParams.perQueue[IGPUObjectFromAssetConverter::EQU_TRANSFER].fence = &engine->gpuTransferFence;
			engine->cpu2gpuParams.perQueue[IGPUObjectFromAssetConverter::EQU_COMPUTE].fence = &engine->gpuComputeFence;
		}

		engine->inputData = new uint32_t[BUFFER_DWORD_COUNT];
		{
			std::mt19937 randGenerator(std::time(0));
			for (uint32_t i = 0u; i < BUFFER_DWORD_COUNT; i++)
				engine->inputData[i] = randGenerator();
		}

		auto gpuinputDataBuffer = engine->utilities->createFilledDeviceLocalGPUBufferOnDedMem(engine->queues[decltype(initOutput)::EQT_TRANSFER_UP], kBufferSize, engine->inputData);

		constexpr const auto totalBufferCount = outputBufferCount + 1u;

		for (auto i = 0; i < outputBufferCount; i++)
		{
			IGPUBuffer::SCreationParams creationParams;
			creationParams.sharingMode = ESM_CONCURRENT;

			IDriverMemoryBacked::SDriverMemoryRequirements reqs;
			reqs.vulkanReqs.memoryTypeBits = ~0u;
			reqs.vulkanReqs.alignment = 256u;
			reqs.vulkanReqs.size = kBufferSize;
			reqs.memoryHeapLocation = IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
			reqs.mappingCapability = IDriverMemoryAllocation::EMCAF_READ;
			engine->buffers[i] = engine->logicalDevice->createGPUBufferOnDedMem(creationParams, reqs, true);
			IDriverMemoryAllocation::MappedMemoryRange mem;
			mem.memory = engine->buffers[i]->getBoundMemory();
			mem.offset = 0u;
			mem.length = kBufferSize;
			engine->logicalDevice->mapMemory(mem, IDriverMemoryAllocation::EMCAF_READ);
		}

		IGPUDescriptorSetLayout::SBinding binding[totalBufferCount];
		for (uint32_t i = 0u; i < totalBufferCount; i++)
			binding[i] = { i,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr };
		auto gpuDSLayout = engine->logicalDevice->createGPUDescriptorSetLayout(binding, binding + totalBufferCount);

		constexpr uint32_t pushconstantSize = 8u * totalBufferCount;
		SPushConstantRange pcRange[1] = { IGPUSpecializedShader::ESS_COMPUTE,0u,pushconstantSize };
		engine->pipelineLayout = engine->logicalDevice->createGPUPipelineLayout(pcRange, pcRange + pushconstantSize, core::smart_refctd_ptr(gpuDSLayout));

		auto descPool = engine->logicalDevice->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, &gpuDSLayout.get(), &gpuDSLayout.get() + 1u);
		engine->descriptorSet = engine->logicalDevice->createGPUDescriptorSet(descPool.get(), core::smart_refctd_ptr(gpuDSLayout));
		{
			IGPUDescriptorSet::SDescriptorInfo infos[totalBufferCount];
			infos[0].desc = gpuinputDataBuffer;
			infos[0].buffer = { 0u,kBufferSize };
			for (uint32_t i = 1u; i <= outputBufferCount; i++)
			{
				infos[i].desc = engine->buffers[i - 1];
				infos[i].buffer = { 0u,kBufferSize };

			}
			IGPUDescriptorSet::SWriteDescriptorSet writes[totalBufferCount];
			for (uint32_t i = 0u; i < totalBufferCount; i++)
				writes[i] = { engine->descriptorSet.get(),i,0u,1u,EDT_STORAGE_BUFFER,infos + i };
			engine->logicalDevice->updateDescriptorSets(totalBufferCount, writes, 0u, nullptr);
		}

		auto getShaderGLSL = [&](const char* filePath) -> auto
		{
			IAssetLoader::SAssetLoadParams lparams;
			lparams.workingDirectory = std::filesystem::current_path();
			auto bundle = engine->assetManager->getAsset(filePath, lparams);
			assert(!bundle.getContents().empty() && bundle.getAssetType() == IAsset::ET_SPECIALIZED_SHADER);
			return core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(*bundle.getContents().begin());
		};

		engine->shaderGLSL[0] = getShaderGLSL("../testSubgroupReduce.comp");
		engine->shaderGLSL[1] = getShaderGLSL("../testSubgroupExclusive.comp");
		engine->shaderGLSL[2] = getShaderGLSL("../testSubgroupInclusive.comp");
		engine->shaderGLSL[3] = getShaderGLSL("../testWorkgroupReduce.comp");
		engine->shaderGLSL[4] = getShaderGLSL("../testWorkgroupExclusive.comp");
		engine->shaderGLSL[5] = getShaderGLSL("../testWorkgroupInclusive.comp");

		
		engine->computeQueue = initOutput.queues[CommonAPI::InitOutput<1>::EQT_COMPUTE];
		engine->fence = engine->logicalDevice->createFence(IGPUFence::ECF_UNSIGNALED);
		auto cmdPool = engine->logicalDevice->createCommandPool(engine->computeQueue->getFamilyIndex(), IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
		engine->logicalDevice->createCommandBuffers(cmdPool.get(), IGPUCommandBuffer::EL_PRIMARY, 1u, &engine->cmdbuf);
	}

	void onAppTerminated_impl(void* data) override
	{
		Nabla* engine = static_cast<Nabla*>(data);

		engine->logger->log("==========Result==========", system::ILogger::ELL_INFO);
		engine->logger->log("Fail Count: %d", system::ILogger::ELL_INFO, engine->totalFailCount);
		delete[] engine->inputData;
	}

	void workLoopBody(void* data) override
	{
		Nabla* engine = static_cast<Nabla*>(data);

		core::smart_refctd_ptr<IGPUComputePipeline> pipelines[Nabla::kTestTypeCount];
		for (uint32_t i = 0u; i < Nabla::kTestTypeCount; i++)
			pipelines[i] = engine->logicalDevice->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(engine->pipelineLayout), std::move(engine->getGPUShader(engine->shaderGLSL[i].get(), engine->workgroupSize)));

		bool passed = true;

		const video::IGPUDescriptorSet* ds = engine->descriptorSet.get();
		passed = runTest<emulatedSubgroupReduction>(engine->logicalDevice.get(), engine->computeQueue, engine->fence.get(), engine->cmdbuf.get(), pipelines[0u].get(), engine->descriptorSet.get(), engine->inputData, engine->workgroupSize, engine->buffers, engine->logger.get()) && passed;
		passed = runTest<emulatedSubgroupScanExclusive>(engine->logicalDevice.get(), engine->computeQueue, engine->fence.get(), engine->cmdbuf.get(), pipelines[1u].get(), engine->descriptorSet.get(), engine->inputData, engine->workgroupSize, engine->buffers, engine->logger.get()) && passed;
		passed = runTest<emulatedSubgroupScanInclusive>(engine->logicalDevice.get(), engine->computeQueue, engine->fence.get(), engine->cmdbuf.get(), pipelines[2u].get(), engine->descriptorSet.get(), engine->inputData, engine->workgroupSize, engine->buffers, engine->logger.get()) && passed;
		passed = runTest<emulatedWorkgroupReduction>(engine->logicalDevice.get(), engine->computeQueue, engine->fence.get(), engine->cmdbuf.get(), pipelines[3u].get(), engine->descriptorSet.get(), engine->inputData, engine->workgroupSize, engine->buffers, engine->logger.get(), true) && passed;
		passed = runTest<emulatedWorkgroupScanExclusive>(engine->logicalDevice.get(), engine->computeQueue, engine->fence.get(), engine->cmdbuf.get(), pipelines[4u].get(), engine->descriptorSet.get(), engine->inputData, engine->workgroupSize, engine->buffers, engine->logger.get(), true) && passed;
		passed = runTest<emulatedWorkgroupScanInclusive>(engine->logicalDevice.get(), engine->computeQueue, engine->fence.get(), engine->cmdbuf.get(), pipelines[5u].get(), engine->descriptorSet.get(), engine->inputData, engine->workgroupSize, engine->buffers, engine->logger.get(), true) && passed;

		if (passed)
			engine->logger->log("Passed test #%d", system::ILogger::ELL_INFO, engine->workgroupSize);
		else
		{
			engine->totalFailCount++;
			engine->logger->log("Failed test #%d", system::ILogger::ELL_ERROR, engine->workgroupSize);
		}

		engine->workgroupSize++;
	}

	bool keepRunning(void* params) override
	{
		Nabla* engine = static_cast<Nabla*>(params);
		return engine->workgroupSize <= 1024u;
	}

};

NBL_COMMON_API_MAIN(ArithmeticUnitTestApp, ArithmeticUnitTestApp::Nabla)
