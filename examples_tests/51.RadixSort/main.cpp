#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "nbl/ext/Scan/Scan.h"
#include "../../source/Nabla/COpenGLDriver.h"

using namespace nbl;
using namespace core;
using namespace video;
using namespace asset;

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

	// Stress Test
	// Todo: Remove this stupid thing before merging

#if 1
	while (true)
	{
#endif
		const size_t in_count = (rand() * 10) + (rand() % 10); //  (1 << 23) - 23u;
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
		auto out_gpu = driver->createDeviceLocalGPUBufferOnDedMem(in_size);

		// Begin Radix Sort

		const int bits_per_pass = 4;
		const int buckets_count = 1 << bits_per_pass;
		int wg_count = (in_count + WG_SIZE - 1) / WG_SIZE;

		const size_t histogram_count = wg_count * buckets_count;
		const size_t histogram_size = histogram_count * sizeof(uint32_t);
		auto histogram_gpu = driver->createDeviceLocalGPUBufferOnDedMem(histogram_size);

		smart_refctd_ptr<IGPUComputePipeline> histogram_pipeline = nullptr;
		smart_refctd_ptr<IGPUDescriptorSet> ds_histogram = nullptr;
		{
			const uint32_t count = 2u;
			IGPUDescriptorSetLayout::SBinding binding[count];
			for (uint32_t i = 0; i < count; ++i)
				binding[i] = { i, asset::EDT_STORAGE_BUFFER, 1u, IGPUSpecializedShader::ESS_COMPUTE, nullptr };

			auto ds_layout_gpu = driver->createGPUDescriptorSetLayout(binding, binding + count);
			ds_histogram = driver->createGPUDescriptorSet(smart_refctd_ptr(ds_layout_gpu));

			auto pipeline_layout = driver->createGPUPipelineLayout(nullptr, nullptr, smart_refctd_ptr(ds_layout_gpu));

			smart_refctd_ptr<IGPUSpecializedShader> shader_gpu = nullptr;
			{
				auto file = smart_refctd_ptr<io::IReadFile>(filesystem->createAndOpenFile("../Histogram.comp"));

				asset::IAssetLoader::SAssetLoadParams lp;
				auto cs_bundle = am->getAsset("../Histogram.comp", lp);
				auto cs = smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*cs_bundle.getContents().begin());
				auto cs_rawptr = cs.get();

				shader_gpu = driver->getGPUObjectsFromAssets(&cs_rawptr, &cs_rawptr + 1)->front();
			}

			histogram_pipeline = driver->createGPUComputePipeline(nullptr, std::move(pipeline_layout), std::move(shader_gpu));
		}

		smart_refctd_ptr<IGPUComputePipeline> scatter_pipeline = nullptr;
		smart_refctd_ptr<IGPUDescriptorSet> ds_scatter = nullptr;
		{
			const uint32_t count = 3u;
			IGPUDescriptorSetLayout::SBinding binding[count];
			for (uint32_t i = 0; i < count; ++i)
				binding[i] = { i, asset::EDT_STORAGE_BUFFER, 1u, IGPUSpecializedShader::ESS_COMPUTE, nullptr };

			auto ds_layout_gpu = driver->createGPUDescriptorSetLayout(binding, binding + count);
			ds_scatter = driver->createGPUDescriptorSet(smart_refctd_ptr(ds_layout_gpu));

			auto pipeline_layout = driver->createGPUPipelineLayout(nullptr, nullptr, smart_refctd_ptr(ds_layout_gpu));

			smart_refctd_ptr<IGPUSpecializedShader> shader_gpu = nullptr;
			{
				auto file = smart_refctd_ptr<io::IReadFile>(filesystem->createAndOpenFile("../Scatter.comp"));

				asset::IAssetLoader::SAssetLoadParams lp;
				auto cs_bundle = am->getAsset("../Scatter.comp", lp);
				auto cs = smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*cs_bundle.getContents().begin());
				auto cs_rawptr = cs.get();

				shader_gpu = driver->getGPUObjectsFromAssets(&cs_rawptr, &cs_rawptr + 1)->front();
			}

			scatter_pipeline = driver->createGPUComputePipeline(nullptr, std::move(pipeline_layout), std::move(shader_gpu));
		}

		core::smart_refctd_ptr<ScanClass> scanner = core::make_smart_refctd_ptr<ScanClass>(driver, ScanClass::Operator::ADD, WG_SIZE);

		auto ds_upsweep = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(scanner->getDefaultDescriptorSetLayout()));
		IGPUComputePipeline* upsweep_pipeline = scanner->getDefaultUpsweepPipeline();

		auto ds_downsweep = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(scanner->getDefaultDescriptorSetLayout()));
		IGPUComputePipeline* downsweep_pipeline = scanner->getDefaultDownsweepPipeline();

		driver->beginScene(true);

		const uint32_t pass_count = 32 / bits_per_pass;

		for (uint32_t pass = 0; pass < pass_count; ++pass)
		{
			{
				const uint32_t count = 2;
				IGPUDescriptorSet::SDescriptorInfo ds_info[count];
				ds_info[0].desc = ((pass % 2) ? out_gpu : in_gpu);
				ds_info[0].buffer = { 0u, in_size };
			
				ds_info[1].desc = histogram_gpu;
				ds_info[1].buffer = { 0u, histogram_size };
			
				IGPUDescriptorSet::SWriteDescriptorSet writes[count];
				for (uint32_t i = 0; i < count; ++i)
					writes[i] = { ds_histogram.get(), i, 0u, 1u, asset::EDT_STORAGE_BUFFER, ds_info + i };
			
				driver->updateDescriptorSets(count, writes, 0u, nullptr);
			}
			
			driver->bindComputePipeline(histogram_pipeline.get());
			driver->bindDescriptorSets(video::EPBP_COMPUTE, histogram_pipeline->getLayout(), 0u, 1u, &ds_histogram.get(), nullptr);
			
			uint32_t shift = pass * 4;
			driver->pushConstants(histogram_pipeline->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(uint32_t), &shift);
			driver->pushConstants(histogram_pipeline->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 4u, sizeof(uint32_t), &in_count);
			driver->dispatch(wg_count, 1, 1);
			
			video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

			{
				ScanClass::Parameters_t scan_push_constants;
				ScanClass::DispatchInfo_t scan_dispatch_info;
				ScanClass::buildParameters(histogram_count, &scan_push_constants, &scan_dispatch_info, WG_SIZE);

				for (uint32_t pass = 0; pass < scan_dispatch_info.upsweep_pass_count; ++pass)
				{
					ScanClass::prePassParameterUpdate(pass, true, &scan_push_constants, &scan_dispatch_info, WG_SIZE);

					ScanClass::updateDescriptorSet(ds_upsweep.get(), histogram_gpu, driver);

					driver->bindComputePipeline(upsweep_pipeline);
					driver->bindDescriptorSets(video::EPBP_COMPUTE, upsweep_pipeline->getLayout(), 0u, 1u, &ds_upsweep.get(), nullptr);
					ScanClass::dispatchHelper(upsweep_pipeline->getLayout(), scan_push_constants, scan_dispatch_info, driver);

					ScanClass::postPassParameterUpdate(pass, true, &scan_push_constants, &scan_dispatch_info, WG_SIZE);
				}

				for (uint32_t pass = 0; pass < scan_dispatch_info.downsweep_pass_count; ++pass)
				{
					ScanClass::prePassParameterUpdate(pass, false, &scan_push_constants, &scan_dispatch_info, WG_SIZE);

					ScanClass::updateDescriptorSet(ds_downsweep.get(), histogram_gpu, driver);

					driver->bindComputePipeline(downsweep_pipeline);
					driver->bindDescriptorSets(video::EPBP_COMPUTE, downsweep_pipeline->getLayout(), 0u, 1u, &ds_downsweep.get(), nullptr);
					ScanClass::dispatchHelper(downsweep_pipeline->getLayout(), scan_push_constants, scan_dispatch_info, driver);

					ScanClass::postPassParameterUpdate(pass, false, &scan_push_constants, &scan_dispatch_info, WG_SIZE);
				}
			}
			
			{
				const uint32_t count = 3;
				IGPUDescriptorSet::SDescriptorInfo ds_info[count];
				ds_info[0].desc = ((pass % 2) ? out_gpu : in_gpu);
				ds_info[0].buffer = { 0u, in_size };
			
				ds_info[1].desc = ((pass % 2) ? in_gpu : out_gpu);
				ds_info[1].buffer = { 0u, in_size };
			
				ds_info[2].desc = histogram_gpu;
				ds_info[2].buffer = { 0u, histogram_size };
			
				IGPUDescriptorSet::SWriteDescriptorSet writes[count];
				for (uint32_t i = 0; i < count; ++i)
					writes[i] = { ds_scatter.get(), i, 0u, 1u, asset::EDT_STORAGE_BUFFER, ds_info + i };
			
				driver->updateDescriptorSets(count, writes, 0u, nullptr);
			}
			
			driver->bindComputePipeline(scatter_pipeline.get());
			driver->bindDescriptorSets(video::EPBP_COMPUTE, scatter_pipeline->getLayout(), 0u, 1u, &ds_scatter.get(),
				nullptr);
			
			driver->pushConstants(scatter_pipeline->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(uint32_t),
				&shift);
			driver->pushConstants(scatter_pipeline->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 4u, sizeof(uint32_t),
				&in_count);
			driver->dispatch(wg_count, 1, 1);
			
			video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		}

		std::cout << "CPU Sort begins" << std::endl;
		std::stable_sort(in, in + in_count, [](const SortElement& a, const SortElement& b) { return a.key < b.key; });
		std::cout << "CPU Sort ends" << std::endl;

		DebugCompareGPUvsCPU<SortElement>(in_gpu, in, in_size, driver);

		driver->endScene();

		delete[] in;
#if 1
	}
#endif

	return 0;
}