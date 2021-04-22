#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "nbl/ext/RadixSort/RadixSort.h"
#include "../../source/Nabla/COpenGLDriver.h"

#include <chrono>
#include <random>

using namespace nbl;
using namespace core;
using namespace video;
using namespace asset;

using RadixSortClass = ext::RadixSort::RadixSort;
using ScanClass = ext::RadixSort::ScanClass;

#define WG_SIZE 256

struct SortElement
{
	uint32_t key, data;

	bool operator!= (const SortElement& other)
	{
		return (key != other.key) || (data != other.data);
	}
};

struct SortElementKeyAccessor
{
	_NBL_STATIC_INLINE_CONSTEXPR size_t key_bit_count = 32ull;

	template<auto bit_offset, auto radix_mask>
	inline decltype(radix_mask) operator()(const SortElement& item) const
	{
		return static_cast<decltype(radix_mask)>(item.key >> static_cast<uint32_t>(bit_offset)) & radix_mask;
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

static void RadixSort(IVideoDriver* driver, const SBufferRange<IGPUBuffer>& in_gpu_range,
	IGPUDescriptorSet* ds_histogram, IGPUComputePipeline* histogram_pipeline,
	IGPUDescriptorSet*  ds_scan, IGPUComputePipeline*  upsweep_pipeline, IGPUComputePipeline* downsweep_pipeline,
	IGPUDescriptorSet* ds_scatter, IGPUComputePipeline* scatter_pipeline)
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

	for (uint32_t pass = 0; pass < RadixSortClass::PASS_COUNT; ++pass)
	{
		// Todo: proper if-else
		(pass == 0u)
			? RadixSortClass::updateDescriptorSet(ds_histogram, { (pass % 2) ? scratch_gpu_range : in_gpu_range, histogram_gpu_range }, driver)
			: RadixSortClass::updateDescriptorSet(ds_histogram, { (pass % 2) ? scratch_gpu_range : in_gpu_range }, driver);

		driver->bindComputePipeline(histogram_pipeline);
		driver->bindDescriptorSets(video::EPBP_COMPUTE, histogram_pipeline->getLayout(), 0u, 1u, &ds_histogram, nullptr);
		RadixSortClass::dispatchHelper(histogram_pipeline->getLayout(), sort_push_constants[pass], sort_dispatch_info, driver);

		for (uint32_t pass = 0; pass < total_scan_pass_count; ++pass)
		{
			ScanClass::updateDescriptorSet(ds_scan, histogram_gpu_range.buffer, driver);
			driver->bindDescriptorSets(video::EPBP_COMPUTE, upsweep_pipeline->getLayout(), 0u, 1u, &ds_scan, nullptr);

			if (pass < upsweep_pass_count)
			{
				driver->bindComputePipeline(upsweep_pipeline);
				ScanClass::dispatchHelper(upsweep_pipeline->getLayout(), scan_push_constants[pass], scan_dispatch_info[pass], driver);
			}
			else
			{
				driver->bindComputePipeline(downsweep_pipeline);
				ScanClass::dispatchHelper(downsweep_pipeline->getLayout(), scan_push_constants[total_scan_pass_count - 1 - pass], scan_dispatch_info[total_scan_pass_count - 1 - pass], driver);
			}
		}

		(pass == 0u)
			? RadixSortClass::updateDescriptorSet(ds_scatter, { (pass % 2) ? scratch_gpu_range : in_gpu_range, (pass % 2) ? in_gpu_range : scratch_gpu_range, histogram_gpu_range }, driver)
			: RadixSortClass::updateDescriptorSet(ds_scatter, { (pass % 2) ? scratch_gpu_range : in_gpu_range, (pass % 2) ? in_gpu_range : scratch_gpu_range }, driver);

		driver->bindComputePipeline(scatter_pipeline);
		driver->bindDescriptorSets(video::EPBP_COMPUTE, scatter_pipeline->getLayout(), 0u, 1u, &ds_scatter, nullptr);
		RadixSortClass::dispatchHelper(scatter_pipeline->getLayout(), sort_push_constants[pass], sort_dispatch_info, driver);
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
	params.StreamingDownloadBufferSize = 0x10000000u; // 256MB download required
	auto device = createDeviceEx(params);

	if (!device)
		return 1;

	IVideoDriver* driver = device->getVideoDriver();

	io::IFileSystem* filesystem = device->getFileSystem();
	asset::IAssetManager* am = device->getAssetManager();

	// Create (an almost) 256MB input buffer
	const size_t in_count = 1024u; // (1 << 25) - 23;
	const size_t in_size = in_count * sizeof(SortElement);

	std::cout << "Input element count: " << in_count << std::endl;

	std::random_device random_device;
	std::mt19937 generator(random_device());
	std::uniform_int_distribution<uint32_t> distribution(0u, ~0u);

	SortElement* in = new SortElement[in_count];
	for (size_t i = 0u; i < in_count; ++i)
	{
		in[i].key = distribution(generator);
		in[i].data = i;
	}
	
	auto in_gpu = driver->createFilledDeviceLocalGPUBufferOnDedMem(in_size, in);
	
	// Take (an almost) 64MB portion from it to sort
	size_t begin = 0u; // (1 << 23) + 112;
	size_t end = 1024u; // (1 << 24) - 77;
	
	assert((begin & (driver->getRequiredSSBOAlignment() - 1ull)) == 0ull);
	
	SBufferRange<IGPUBuffer> in_gpu_range = { 0 };
	in_gpu_range.offset = begin * sizeof(SortElement);
	in_gpu_range.size = (end - begin) * sizeof(SortElement);
	in_gpu_range.buffer = in_gpu;
	
	auto sorter = core::make_smart_refctd_ptr<RadixSortClass>(driver, WG_SIZE);
	
	// Todo: compress
	auto ds_histogram = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const video::IGPUDescriptorSetLayout>(sorter->getDefaultSortDescriptorSetLayout()));
	auto histogram_pipeline = sorter->getDefaultHistogramPipeline();
	
	auto ds_scan = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const video::IGPUDescriptorSetLayout>(sorter->getDefaultScanDescriptorSetLayout()));
	auto upsweep_pipeline = sorter->getDefaultUpsweepPipeline();
	auto downsweep_pipeline = sorter->getDefaultDownsweepPipeline();
	
	auto ds_scatter = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const video::IGPUDescriptorSetLayout>(sorter->getDefaultSortDescriptorSetLayout()));
	auto scatter_pipeline = sorter->getDefaultScatterPipeline();
	
	{
		driver->beginScene(true);
	
		std::cout << "GPU sort begin" << std::endl;
	
		// Todo: Do proper GPU profiling
		auto start = std::chrono::high_resolution_clock::now();
	
		RadixSort(driver, in_gpu_range, ds_histogram.get(), histogram_pipeline, ds_scan.get(), upsweep_pipeline, downsweep_pipeline, ds_scatter.get(), scatter_pipeline);
	
		auto stop = std::chrono::high_resolution_clock::now();
	
		std::cout << "GPU sort end\nTime taken: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " microseconds" << std::endl;
	
		driver->endScene();
	}

	{	
		std::cout << "CPU sort begin" << std::endl;

		SortElement* in_data = new SortElement[in_count + (end - begin)];
		memcpy(in_data, in, sizeof(SortElement) * in_count);

		auto start = std::chrono::high_resolution_clock::now();
		SortElement* sorted_data = core::radix_sort(in_data + begin, in_data + in_count, end - begin, SortElementKeyAccessor());
		auto stop = std::chrono::high_resolution_clock::now();

		memcpy(in_data + begin, sorted_data, (end - begin) * sizeof(SortElement));

		std::cout << "CPU sort end\nTime taken: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " microseconds" << std::endl;

		std::cout << "Testing: ";
		DebugCompareGPUvsCPU<SortElement>(in_gpu, in_data, in_size, driver);

		delete[] in_data;
	}

	delete[] in;

	return 0;
}