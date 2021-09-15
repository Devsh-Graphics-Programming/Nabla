#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"

#include <chrono>
#include <random>

using namespace nbl;
using namespace core;
using namespace video;
using namespace asset;


#if 0 
template <typename T>
static T* DebugGPUBufferDownload(smart_refctd_ptr<IGPUBuffer> buffer_to_download, size_t buffer_size, IVideoDriver* driver)
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

	driver->copyBuffer(buffer_to_download.get(), downBuffer, 0, address, array_size_32);

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

template <typename T>
static void DebugCompareGPUvsCPU(smart_refctd_ptr<IGPUBuffer> gpu_buffer, T* cpu_buffer, size_t buffer_size, IVideoDriver* driver)
{
	T* downloaded_buffer = DebugGPUBufferDownload<T>(gpu_buffer, buffer_size, driver);

	size_t buffer_count = buffer_size / sizeof(T);

	if (downloaded_buffer)
	{
		for (int i = 0; i < buffer_count; ++i)
		{
			if (downloaded_buffer[i] != cpu_buffer[i])
				__debugbreak();
		}

		std::cout << "PASS" << std::endl;
	}
}

#endif

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
	constexpr auto begin = in_count/4+112;
	constexpr auto end = in_count*3/4-77;
	constexpr auto elementCount = end-begin;
	
	assert((begin&(gpuPhysicalDevice->getLimits().SSBOAlignment-1u))==0u);
	SBufferRange<IGPUBuffer> in_gpu_range = {0};
	in_gpu_range.offset = begin*sizeof(uint32_t);
	in_gpu_range.size = elementCount*sizeof(uint32_t);
	in_gpu_range.buffer = utilities->createFilledDeviceLocalGPUBufferOnDedMem(queues[decltype(initOutput)::EQT_TRANSFER_UP],in_count*sizeof(uint32_t),in);
	
	auto scanner = utilities->getDefaultScanner();
	auto scan_pipeline = scanner->getDefaultPipeline(CScanner::EDT_UINT,CScanner::EO_ADD);

	CScanner::Parameters scan_push_constants;
	CScanner::DispatchInfo scan_dispatch_info;
	scanner->buildParameters(elementCount,scan_push_constants,scan_dispatch_info);

	auto dsLayout = scanner->getDefaultDescriptorSetLayout();
	auto dsPool = logicalDevice->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE,&dsLayout,&dsLayout+1u);
	auto ds = logicalDevice->createGPUDescriptorSet(dsPool.get(),core::smart_refctd_ptr<IGPUDescriptorSetLayout>(dsLayout));
	//scanner->updateDescriptorSet(set,in_gpu_range,scratch_range);

	constexpr auto BenchmarkingRuns = 1000u;
	for (auto i=0u; i<BenchmarkingRuns; i++)
	{
	}
#if 0
	
	RadixSort(driver, in_gpu_range, ds_sort, ds_sort_count, histogram_pipeline, scatter_pipeline, ds_scan.get(), upsweep_pipeline, downsweep_pipeline);
static void RadixSort(IVideoDriver* driver, const SBufferRange<IGPUBuffer>& in_gpu_range,
	core::smart_refctd_ptr<IGPUDescriptorSet>* ds_sort, const uint32_t ds_sort_count,
	IGPUComputePipeline* histogram_pipeline, IGPUComputePipeline* scatter_pipeline,
	IGPUDescriptorSet* ds_scan, IGPUComputePipeline* upsweep_pipeline, IGPUComputePipeline* downsweep_pipeline)
{
	const uint32_t total_scan_pass_count = RadixSortClass::buildParameters(in_gpu_range.size / sizeof(SortElement), WG_SIZE,
		nullptr, nullptr, nullptr, nullptr);

	RadixSortClass::Parameters_t sort_push_constants[RadixSortClass::PASS_COUNT];
	RadixSortClass::DispatchInfo_t sort_dispatch_info;

	const uint32_t upsweep_pass_count = (total_scan_pass_count / 2) + 1;
	core::vector<ScanClass::Parameters_t> scan_push_constants(upsweep_pass_count);
	core::vector<ScanClass::DispatchInfo_t> scan_dispatch_info(upsweep_pass_count);

	RadixSortClass::buildParameters(in_gpu_range.size / sizeof(SortElement), WG_SIZE, sort_push_constants, &sort_dispatch_info,
		scan_push_constants.data(), scan_dispatch_info.data());

	SBufferRange<IGPUBuffer> scratch_gpu_range = { 0 };
	scratch_gpu_range.size = in_gpu_range.size;
	scratch_gpu_range.buffer = driver->createDeviceLocalGPUBufferOnDedMem(in_gpu_range.size);

	const uint32_t histogram_count = sort_dispatch_info.wg_count[0] * RadixSortClass::BUCKETS_COUNT;
	SBufferRange<IGPUBuffer> histogram_gpu_range = { 0 };
	histogram_gpu_range.size = histogram_count * sizeof(uint32_t);
	histogram_gpu_range.buffer = driver->createDeviceLocalGPUBufferOnDedMem(histogram_gpu_range.size);

	RadixSortClass::updateDescriptorSet(ds_scan, &histogram_gpu_range, 1u, driver);
	RadixSortClass::updateDescriptorSetsPingPong(ds_sort, in_gpu_range, scratch_gpu_range, driver);

	core::smart_refctd_ptr<video::IQueryObject> time_query(driver->createElapsedTimeQuery());

	std::cout << "GPU sort begin" << std::endl;

	driver->beginQuery(time_query.get());
	RadixSortClass::sort(histogram_pipeline, upsweep_pipeline, downsweep_pipeline, scatter_pipeline, ds_scan, ds_sort, scan_push_constants.data(),
		sort_push_constants, scan_dispatch_info.data(), &sort_dispatch_info, total_scan_pass_count, upsweep_pass_count, driver);
	driver->endQuery(time_query.get());

	uint32_t time_taken;
	time_query->getQueryResult(&time_taken);

	std::cout << "GPU sort end\nTime taken: " << (double)time_taken / 1000000.0 << " ms" << std::endl;
}
#endif

	for (auto i=0u; i<BenchmarkingRuns; i++)
	{
		logger->log("CPU scan begin",system::ILogger::ELL_PERFORMANCE);

		auto dst = in+begin;
		auto start = std::chrono::high_resolution_clock::now();
		std::inclusive_scan(in+begin,in+end,dst);
		auto stop = std::chrono::high_resolution_clock::now();

		logger->log("CPU sort end. Time taken: %d us",system::ILogger::ELL_PERFORMANCE,std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count());
	}
	//DebugCompareGPUvsCPU(in_gpu,in_data,in_size,driver);

	delete[] in;
	return 0;
}