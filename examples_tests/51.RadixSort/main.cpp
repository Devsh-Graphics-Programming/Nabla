#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "nbl/ext/Scan/Scan.h"
#include "nbl/ext/RadixSort/RadixSort.h"
#include "../../source/Nabla/COpenGLDriver.h"

#include <chrono>

using namespace nbl;
using namespace core;
using namespace video;
using namespace asset;

using RadixSortClass = ext::RadixSort::RadixSort;
using ScanClass = ext::Scan::Scan<uint32_t>;

#define WG_SIZE 256

struct SortElement
{
	uint32_t key, data;

	bool operator!= (const SortElement& other)
	{
		return (key != other.key) || (data != other.data);
	}
};

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

static void ExclusiveSumScan(IVideoDriver* driver, smart_refctd_ptr<IGPUBuffer> in_gpu, const uint32_t in_count, IGPUDescriptorSet* ds_upsweep,
	IGPUComputePipeline* upsweep_pipeline, IGPUDescriptorSet* ds_downsweep, IGPUComputePipeline* downsweep_pipeline)
{
	ScanClass::Parameters_t scan_push_constants;
	ScanClass::DispatchInfo_t scan_dispatch_info;
	ScanClass::buildParameters(in_count, &scan_push_constants, &scan_dispatch_info, WG_SIZE);

	for (uint32_t pass = 0; pass < scan_dispatch_info.upsweep_pass_count; ++pass)
	{
		ScanClass::prePassParameterUpdate(pass, true, &scan_push_constants, &scan_dispatch_info, WG_SIZE);

		ScanClass::updateDescriptorSet(ds_upsweep, in_gpu, driver);

		driver->bindComputePipeline(upsweep_pipeline);
		driver->bindDescriptorSets(video::EPBP_COMPUTE, upsweep_pipeline->getLayout(), 0u, 1u, &ds_upsweep, nullptr);
		ScanClass::dispatchHelper(upsweep_pipeline->getLayout(), scan_push_constants, scan_dispatch_info, driver);

		ScanClass::postPassParameterUpdate(pass, true, &scan_push_constants, &scan_dispatch_info, WG_SIZE);
	}

	for (uint32_t pass = 0; pass < scan_dispatch_info.downsweep_pass_count; ++pass)
	{
		ScanClass::prePassParameterUpdate(pass, false, &scan_push_constants, &scan_dispatch_info, WG_SIZE);

		ScanClass::updateDescriptorSet(ds_downsweep, in_gpu, driver);

		driver->bindComputePipeline(downsweep_pipeline);
		driver->bindDescriptorSets(video::EPBP_COMPUTE, downsweep_pipeline->getLayout(), 0u, 1u, &ds_downsweep, nullptr);
		ScanClass::dispatchHelper(downsweep_pipeline->getLayout(), scan_push_constants, scan_dispatch_info, driver);

		ScanClass::postPassParameterUpdate(pass, false, &scan_push_constants, &scan_dispatch_info, WG_SIZE);
	}
}

static void RadixSort(IVideoDriver* driver,
	smart_refctd_ptr<IGPUBuffer> in_gpu, const uint32_t in_count, smart_refctd_ptr<IGPUBuffer> scratch_gpu,
	smart_refctd_ptr<IGPUBuffer> histogram_gpu,
	IGPUDescriptorSet* ds_histogram, IGPUComputePipeline* histogram_pipeline,
	IGPUDescriptorSet* ds_upsweep, IGPUComputePipeline* upsweep_pipeline,
	IGPUDescriptorSet* ds_downsweep, IGPUComputePipeline* downsweep_pipeline,
	IGPUDescriptorSet* ds_scatter, IGPUComputePipeline* scatter_pipeline,
	RadixSortClass::Parameters_t* radix_sort_push_constants, RadixSortClass::DispatchInfo_t* radix_sort_dispatch_info)
{
	for (uint32_t pass = 0; pass < RadixSortClass::PASS_COUNT; ++pass)
	{
		RadixSortClass::updateParameters(radix_sort_push_constants, pass);

		(pass == 0u)
			? RadixSortClass::updateDescriptorSet(ds_histogram, { (pass % 2) ? scratch_gpu : in_gpu, histogram_gpu }, driver)
			: RadixSortClass::updateDescriptorSet(ds_histogram, { (pass % 2) ? scratch_gpu : in_gpu }, driver);

		driver->bindComputePipeline(histogram_pipeline);
		driver->bindDescriptorSets(video::EPBP_COMPUTE, histogram_pipeline->getLayout(), 0u, 1u, &ds_histogram, nullptr);
		RadixSortClass::dispatchHelper(histogram_pipeline->getLayout(), *radix_sort_push_constants, *radix_sort_dispatch_info, driver);

		ExclusiveSumScan(driver, histogram_gpu, radix_sort_dispatch_info->histogram_count, ds_upsweep, upsweep_pipeline,
			ds_downsweep, downsweep_pipeline);

		(pass == 0u)
			? RadixSortClass::updateDescriptorSet(ds_scatter, { (pass % 2) ? scratch_gpu : in_gpu, (pass % 2) ? in_gpu : scratch_gpu, histogram_gpu }, driver)
			: RadixSortClass::updateDescriptorSet(ds_scatter, { (pass % 2) ? scratch_gpu : in_gpu, (pass % 2) ? in_gpu : scratch_gpu }, driver);

		driver->bindComputePipeline(scatter_pipeline);
		driver->bindDescriptorSets(video::EPBP_COMPUTE, scatter_pipeline->getLayout(), 0u, 1u, &ds_scatter, nullptr);
		RadixSortClass::dispatchHelper(scatter_pipeline->getLayout(), *radix_sort_push_constants, *radix_sort_dispatch_info, driver);
	}
}

int main()
{
	nbl::SIrrlichtCreationParameters params;
	params.Bits = 24;
	params.ZBufferBits = 24;
	params.DriverType = video::EDT_OPENGL;
	params.WindowSize = dimension2d<uint32_t>(512, 512);
	params.Fullscreen = false;
	params.Vsync = true;
	params.Doublebuffer = true;
	params.Stencilbuffer = false;
	auto device = createDeviceEx(params);

	if (!device)
		return 1;

	IVideoDriver* driver = device->getVideoDriver();

	io::IFileSystem* filesystem = device->getFileSystem();
	asset::IAssetManager* am = device->getAssetManager();

	unsigned int seed = 128;

	const size_t in_count = (1 << 23) - 23;
	const size_t in_size = in_count * sizeof(SortElement);

	std::cout << "Input element count: " << in_count << std::endl;

	SortElement* in = new SortElement[in_count];
	srand(seed++);
	for (size_t i = 0u; i < in_count; ++i)
	{
		in[i].key = rand();
		in[i].data = i;
	}
	
	auto in_gpu = driver->createFilledDeviceLocalGPUBufferOnDedMem(in_size, in);

	auto sorter = core::make_smart_refctd_ptr<RadixSortClass>(driver, WG_SIZE);
	auto scanner = core::make_smart_refctd_ptr<ScanClass>(driver, ScanClass::Operator::ADD, WG_SIZE);

	auto ds_histogram = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const video::IGPUDescriptorSetLayout>(
		sorter->getDefaultHistogramDescriptorSetLayout()));
	auto histogram_pipeline = sorter->getDefaultHistogramPipeline();

	auto ds_upsweep = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const video::IGPUDescriptorSetLayout>(
		scanner->getDefaultDescriptorSetLayout()));
	auto upsweep_pipeline = scanner->getDefaultUpsweepPipeline();

	auto ds_downsweep = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const video::IGPUDescriptorSetLayout>(
		scanner->getDefaultDescriptorSetLayout()));
	auto downsweep_pipeline = scanner->getDefaultDownsweepPipeline();

	auto ds_scatter = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const video::IGPUDescriptorSetLayout>(
		sorter->getDefaultScatterDescriptorSetLayout()));
	auto scatter_pipeline = sorter->getDefaultScatterPipeline();

	RadixSortClass::Parameters_t radix_sort_push_constants;
	RadixSortClass::DispatchInfo_t radix_sort_dispatch_info;
	RadixSortClass::buildParameters(in_count, &radix_sort_push_constants, &radix_sort_dispatch_info, WG_SIZE);

	auto scratch_gpu = driver->createDeviceLocalGPUBufferOnDedMem(in_gpu->getSize());
	auto histogram_gpu = driver->createDeviceLocalGPUBufferOnDedMem(radix_sort_dispatch_info.histogram_count * sizeof(uint32_t));

	{
		driver->beginScene(true);

		std::cout << "GPU sort begin" << std::endl;
		auto begin = std::chrono::high_resolution_clock::now();

		RadixSort(driver, in_gpu, in_count, scratch_gpu, histogram_gpu, ds_histogram.get(), histogram_pipeline,
			ds_upsweep.get(), upsweep_pipeline, ds_downsweep.get(), downsweep_pipeline, ds_scatter.get(), scatter_pipeline,
			&radix_sort_push_constants, &radix_sort_dispatch_info);

		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "GPU sort end" << std::endl;

		std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " microseconds" << std::endl;

		driver->endScene();
	}

	{
		std::cout << "CPU sort begin" << std::endl;
		auto begin = std::chrono::high_resolution_clock::now();

		std::stable_sort(in, in + in_count, [](const SortElement& a, const SortElement& b) { return a.key < b.key; });

		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "CPU sort end" << std::endl;

		std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " microseconds" << std::endl;
	}

	std::cout << "Testing: ";
	DebugCompareGPUvsCPU<SortElement>(in_gpu, in, in_size, driver);

	delete[] in;

	return 0;
}