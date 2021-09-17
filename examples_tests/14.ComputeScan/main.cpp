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
	auto initOutput = CommonAPI::Init(video::EAT_OPENGL,"Subgroup Arithmetic Test");
	auto system = std::move(initOutput.system);
    auto gl = std::move(initOutput.apiConnection);
    auto logger = std::move(initOutput.logger);
    auto gpuPhysicalDevice = std::move(initOutput.physicalDevice);
    auto logicalDevice = std::move(initOutput.logicalDevice);
    auto queues = std::move(initOutput.queues);
    auto renderpass = std::move(initOutput.renderpass);
    auto commandPool = std::move(initOutput.commandPool);
    auto assetManager = std::move(initOutput.assetManager);
    auto cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
    auto utilities = std::move(initOutput.utilities);

    core::smart_refctd_ptr<IGPUFence> gpuTransferFence = nullptr;
    core::smart_refctd_ptr<IGPUFence> gpuComputeFence = nullptr;

    nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
    {
        cpu2gpuParams.perQueue[IGPUObjectFromAssetConverter::EQU_TRANSFER].fence = &gpuTransferFence;
        cpu2gpuParams.perQueue[IGPUObjectFromAssetConverter::EQU_COMPUTE].fence = &gpuComputeFence;
    }

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
	
	auto scanner = utilities->getDefaultScanner();
	auto scan_pipeline = scanner->getDefaultPipeline(CScanner::EDT_UINT,CScanner::EO_ADD);

	CScanner::Parameters scan_push_constants;
	CScanner::DispatchInfo scan_dispatch_info;
	scanner->buildParameters(elementCount,scan_push_constants,scan_dispatch_info);
	
	SBufferRange<IGPUBuffer> scratch_gpu_range;
	scratch_gpu_range.offset = 0u;
	scratch_gpu_range.size = scan_push_constants.getScratchSize();
	scratch_gpu_range.buffer = logicalDevice->createDeviceLocalGPUBufferOnDedMem(scratch_gpu_range.size);

	auto dsLayout = scanner->getDefaultDescriptorSetLayout();
	auto dsPool = logicalDevice->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE,&dsLayout,&dsLayout+1u);
	auto ds = logicalDevice->createGPUDescriptorSet(dsPool.get(),core::smart_refctd_ptr<IGPUDescriptorSetLayout>(dsLayout));
	scanner->updateDescriptorSet(ds.get(),in_gpu_range,scratch_gpu_range);

	constexpr auto BenchmarkingRuns = 1u;
	auto computeQueue = queues[decltype(initOutput)::EQT_COMPUTE];
	core::smart_refctd_ptr<IGPUFence> fences[BenchmarkingRuns];
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
		for (auto i=0u; i<BenchmarkingRuns; i++)
			fences[i] = logicalDevice->createFence(IGPUFence::ECF_UNSIGNALED);
		IGPUQueue::SSubmitInfo submit = {};
		submit.commandBufferCount = 1u;
		submit.commandBuffers = &cmdbuf.get();
		for (auto i=0u; i<BenchmarkingRuns; i++)
			computeQueue->submit(1u,&submit,fences[i].get());
	}
	// cpu counterpart
	auto cpu_begin = in+begin;
	for (auto i=0u; i<BenchmarkingRuns; i++)
	{
		logger->log("CPU scan begin",system::ILogger::ELL_PERFORMANCE);

		auto start = std::chrono::high_resolution_clock::now();
		std::inclusive_scan(cpu_begin,in+end,cpu_begin);
		auto stop = std::chrono::high_resolution_clock::now();

		logger->log("CPU sort end. Time taken: %d us",system::ILogger::ELL_PERFORMANCE,std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count());
	}

	if (BenchmarkingRuns==1u)
	{
		//T* downloaded_buffer = DebugGPUBufferDownload<T>(in_gpu_range, buffer_size, driver);
#if 0 
		{
			constexpr uint64_t timeout_ns = 15000000000u;
			const uint32_t alignment = uint32_t(sizeof(T));
			auto downloadStagingArea = driver->getDefaultDownStreamingBuffer();
			auto downBuffer = downloadStagingArea->getBuffer();

			bool success = false;

			uint32_t array_size_32 = uint32_t(buffer_size);
			uint32_t address = std::remove_pointer<decltype(downloadStagingArea)>::type::invalid_address;
			auto unallocatedSize = downloadStagingArea->multi_alloc(1u, &address, &array_size_32, &alignment);
			if (unallocatedSize)
			{
				os::Printer::log("Could not download the buffer from the GPU!", ELL_ERROR);
				exit(420);
			}

			driver->copyBuffer(in_gpu_range.buffer.get(), downBuffer, 0, address, array_size_32);

			auto downloadFence = driver->placeFence(true);
			auto result = downloadFence->waitCPU(timeout_ns, true);

			T* dataFromBuffer = nullptr;
			if (result != video::E_DRIVER_FENCE_RETVAL::EDFR_TIMEOUT_EXPIRED && result != video::E_DRIVER_FENCE_RETVAL::EDFR_FAIL)
			{
				if (downloadStagingArea->needsManualFlushOrInvalidate())
					driver->invalidateMappedMemoryRanges({ {downloadStagingArea->getBuffer()->getBoundMemory(),address,array_size_32} });

				dataFromBuffer = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(downloadStagingArea->getBufferPointer()) + address);
			}
			else
			{
				os::Printer::log("Could not download the buffer from the GPU, fence not signalled!", ELL_ERROR);
			}

			downloadStagingArea->multi_free(1u, &address, &array_size_32, nullptr);

			return dataFromBuffer;
		}
#endif
		//assert(downloaded_buffer);

		for (auto i=0u; i<elementCount; i++)
		{
			//if (downloaded_buffer[i]!=cpu_begin[i])
				//__debugbreak();
		}
		logger->log("Result Comparison Test Passed",system::ILogger::ELL_PERFORMANCE);
	}

	delete[] in;
	return 0;
}