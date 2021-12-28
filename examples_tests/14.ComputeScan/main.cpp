#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"

#include <chrono>
#include <random>

using namespace nbl;
using namespace core;
using namespace video;
using namespace asset;


int main()
{
	CommonAPI::SFeatureRequest<video::IAPIConnection::E_FEATURE> requiredInstanceFeatures = {};
	CommonAPI::SFeatureRequest<video::IAPIConnection::E_FEATURE> optionalInstanceFeatures = {};
	CommonAPI::SFeatureRequest<video::ILogicalDevice::E_FEATURE> requiredDeviceFeatures = {};
	CommonAPI::SFeatureRequest< video::ILogicalDevice::E_FEATURE> optionalDeviceFeatures = {};

	auto initOutput = CommonAPI::Init(video::EAT_OPENGL, "Subgroup Arithmetic Test", requiredInstanceFeatures, optionalInstanceFeatures, requiredDeviceFeatures, optionalDeviceFeatures);
	auto system = std::move(initOutput.system);
	auto gl = std::move(initOutput.apiConnection);
	auto logger = std::move(initOutput.logger);
	auto gpuPhysicalDevice = std::move(initOutput.physicalDevice);
	auto logicalDevice = std::move(initOutput.logicalDevice);
	auto queues = std::move(initOutput.queues);
	auto renderpass = std::move(initOutput.renderpass);
	auto computeCommandPool = std::move(initOutput.commandPools[CommonAPI::InitOutput::EQT_COMPUTE]);
	auto commandPool = computeCommandPool;
	auto assetManager = std::move(initOutput.assetManager);
	auto cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
	auto utilities = std::move(initOutput.utilities);

	core::smart_refctd_ptr<IGPUFence> gpuTransferFence = nullptr;
	core::smart_refctd_ptr<IGPUFence> gpuComputeFence = nullptr;

	nbl::video::IGPUObjectFromAssetConverter cpu2gpu;

	// Create (an almost) 128MB input buffer
	constexpr auto in_size = 128u<<20u;
	constexpr auto in_count = in_size/sizeof(uint32_t)-23u;

	logger->log("Input element count: %d",system::ILogger::ELL_PERFORMANCE,in_count);

	auto in = new uint32_t[in_count];
	{
		std::random_device random_device;
		std::mt19937 generator(random_device());
		std::uniform_int_distribution<uint32_t> distribution(0u, ~0u);
		for (auto i=0u; i<in_count; i++)
			in[i] = distribution(generator);
	}
	
	// Take (an almost) 64MB portion from it to scan
	constexpr auto begin = in_count/4+110;
	assert(((begin*sizeof(uint32_t))&(gpuPhysicalDevice->getLimits().SSBOAlignment-1u))==0u);
	constexpr auto end = in_count*3/4-78;
	assert(((end*sizeof(uint32_t))&(gpuPhysicalDevice->getLimits().SSBOAlignment-1u))==0u);
	constexpr auto elementCount = end-begin;
	
	SBufferRange<IGPUBuffer> in_gpu_range;
	in_gpu_range.offset = begin*sizeof(uint32_t);
	in_gpu_range.size = elementCount*sizeof(uint32_t);
	in_gpu_range.buffer = utilities->createFilledDeviceLocalGPUBufferOnDedMem(queues[decltype(initOutput)::EQT_TRANSFER_UP],in_count*sizeof(uint32_t),in);
	
	const auto scanType = video::CScanner::EST_EXCLUSIVE;
	auto scanner = utilities->getDefaultScanner();
	auto scan_pipeline = scanner->getDefaultPipeline(scanType,CScanner::EDT_UINT,CScanner::EO_ADD);

	CScanner::DefaultPushConstants scan_push_constants;
	CScanner::DispatchInfo scan_dispatch_info;
	scanner->buildParameters(elementCount,scan_push_constants,scan_dispatch_info);
	
	SBufferRange<IGPUBuffer> scratch_gpu_range;
	{
		scratch_gpu_range.offset = 0u;
		scratch_gpu_range.size = scan_push_constants.scanParams.getScratchSize();
		IGPUBuffer::SCreationParams params = {};
		params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
		scratch_gpu_range.buffer = logicalDevice->createDeviceLocalGPUBufferOnDedMem(params,scratch_gpu_range.size);
	}

	auto dsLayout = scanner->getDefaultDescriptorSetLayout();
	auto dsPool = logicalDevice->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE,&dsLayout,&dsLayout+1u);
	auto ds = logicalDevice->createGPUDescriptorSet(dsPool.get(),core::smart_refctd_ptr<IGPUDescriptorSetLayout>(dsLayout));
	scanner->updateDescriptorSet(ds.get(),in_gpu_range,scratch_gpu_range);

	constexpr auto BenchmarkingRuns = 1u;
	auto computeQueue = queues[decltype(initOutput)::EQT_COMPUTE];
	core::smart_refctd_ptr<IGPUFence> lastFence;
	// TODO: timestamp queries
	//core::smart_refctd_ptr<> beginTimestamp[BenchmarkingRuns];
	//core::smart_refctd_ptr<> endTimestamp[BenchmarkingRuns];
	{
		core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
		{
			auto cmdPool = logicalDevice->createCommandPool(computeQueue->getFamilyIndex(),IGPUCommandPool::ECF_NONE);
			logicalDevice->createCommandBuffers(cmdPool.get(),IGPUCommandBuffer::EL_PRIMARY,1u,&cmdbuf);

			// TODO: barriers
			IGPUCommandBuffer::SBufferMemoryBarrier srcBufferBarrier;
			IGPUCommandBuffer::SBufferMemoryBarrier dstBufferBarrier;

			// TODO: begin and end query
			cmdbuf->begin(IGPUCommandBuffer::EU_SIMULTANEOUS_USE_BIT);
			cmdbuf->fillBuffer(scratch_gpu_range.buffer.get(),0u,sizeof(uint32_t)+scratch_gpu_range.size/2u,0u);
			cmdbuf->bindComputePipeline(scan_pipeline);
			auto pipeline_layout = scan_pipeline->getLayout();
			cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE,pipeline_layout,0u,1u,&ds.get());
			scanner->dispatchHelper(
				cmdbuf.get(),pipeline_layout,scan_push_constants,scan_dispatch_info,
				static_cast<asset::E_PIPELINE_STAGE_FLAGS>(asset::EPSF_COMPUTE_SHADER_BIT|asset::EPSF_TRANSFER_BIT),1u,&srcBufferBarrier,
				static_cast<asset::E_PIPELINE_STAGE_FLAGS>(asset::EPSF_COMPUTE_SHADER_BIT|asset::EPSF_TRANSFER_BIT),1u,&dstBufferBarrier
			);
			cmdbuf->end();
		}
		lastFence = logicalDevice->createFence(IGPUFence::ECF_UNSIGNALED);
		IGPUQueue::SSubmitInfo submit = {};
		submit.commandBufferCount = 1u;
		submit.commandBuffers = &cmdbuf.get();
		computeQueue->startCapture();
		for (auto i=0u; i<BenchmarkingRuns; i++)
			computeQueue->submit(1u,&submit,i!=(BenchmarkingRuns-1u) ? nullptr:lastFence.get());
		computeQueue->endCapture();
	}
	// cpu counterpart
	auto cpu_begin = in+begin;
	for (auto i=0u; i<BenchmarkingRuns; i++)
	{
		logger->log("CPU scan begin",system::ILogger::ELL_PERFORMANCE);

		auto start = std::chrono::high_resolution_clock::now();
		switch (scanType)
		{
			case video::CScanner::EST_INCLUSIVE:
				std::inclusive_scan(cpu_begin,in+end,cpu_begin);
				break;
			case video::CScanner::EST_EXCLUSIVE:
				std::exclusive_scan(cpu_begin,in+end,cpu_begin,0u);
				break;
			default:
				assert(false);
				exit(0xdeadbeefu);
				break;
		}
		auto stop = std::chrono::high_resolution_clock::now();

		logger->log("CPU scan end. Time taken: %d us",system::ILogger::ELL_PERFORMANCE,std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count());
	}
	
	// wait for the gpu impl to complete
	logicalDevice->blockForFences(1u,&lastFence.get());

	if (BenchmarkingRuns==1u)
	{
		IGPUBuffer::SCreationParams params = {};
		params.usage = IGPUBuffer::EUF_TRANSFER_DST_BIT;
		auto downloaded_buffer = logicalDevice->createDownStreamingGPUBufferOnDedMem(params,in_gpu_range.size);
		{
			core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
			{
				auto cmdPool = logicalDevice->createCommandPool(computeQueue->getFamilyIndex(),IGPUCommandPool::ECF_NONE);
				logicalDevice->createCommandBuffers(cmdPool.get(),IGPUCommandBuffer::EL_PRIMARY,1u,&cmdbuf);
			}
			cmdbuf->begin(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
			asset::SBufferCopy region;
			region.srcOffset = in_gpu_range.offset;
			region.dstOffset = 0u;
			region.size = in_gpu_range.size;
			cmdbuf->copyBuffer(in_gpu_range.buffer.get(),downloaded_buffer.get(),1u,&region);
			cmdbuf->end();
			lastFence = logicalDevice->createFence(IGPUFence::ECF_UNSIGNALED);
			IGPUQueue::SSubmitInfo submit = {};
			submit.commandBufferCount = 1u;
			submit.commandBuffers = &cmdbuf.get();
			computeQueue->submit(1u,&submit,lastFence.get());
			logicalDevice->blockForFences(1u,&lastFence.get());
		}

		auto mem = const_cast<video::IDriverMemoryAllocation*>(downloaded_buffer->getBoundMemory());
		{
			video::IDriverMemoryAllocation::MappedMemoryRange range;
			{
				range.memory = mem;
				range.offset = 0u;
				range.length = in_gpu_range.size;
			}
			logicalDevice->mapMemory(range,video::IDriverMemoryAllocation::EMCAF_READ);
		}
		auto gpu_begin = reinterpret_cast<uint32_t*>(mem->getMappedPointer());
		for (auto i=0u; i<elementCount; i++)
		{
			if (gpu_begin[i]!=cpu_begin[i])
				_NBL_DEBUG_BREAK_IF(true);
		}
		logger->log("Result Comparison Test Passed",system::ILogger::ELL_PERFORMANCE);
	}

	delete[] in;
	return 0;
}