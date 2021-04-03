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

#if 1
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

	unsigned int seed = 666;

	const size_t in_count = (rand() * 10) + (rand() % 10);
	const size_t in_size = in_count * sizeof(uint32_t);

	uint32_t* in = new uint32_t[in_count];
	srand(seed++);
	for (size_t i = 0u; i < in_count; ++i)
		in[i] = rand() * std::pow(-1, (i + 1));

	auto in_gpu = driver->createFilledDeviceLocalGPUBufferOnDedMem(in_size, in);

	core::smart_refctd_ptr<ScanClass> scanner = core::make_smart_refctd_ptr<ScanClass>(driver, ScanClass::ADD, WG_SIZE);

	core::smart_refctd_ptr<IGPUDescriptorSet> ds_upsweep = driver->createGPUDescriptorSet(
		core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(scanner->getDefaultDescriptorSetLayout()));
	ScanClass::updateDescriptorSet(ds_upsweep.get(), in_gpu, driver);
	core::smart_refctd_ptr<IGPUComputePipeline> upsweep_pipeline(scanner->getDefaultUpsweepPipeline());

	core::smart_refctd_ptr<IGPUDescriptorSet> ds_downsweep = driver->createGPUDescriptorSet(
		core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(scanner->getDefaultDescriptorSetLayout()));
	ScanClass::updateDescriptorSet(ds_downsweep.get(), in_gpu, driver);
	core::smart_refctd_ptr<IGPUComputePipeline> downsweep_pipeline(scanner->getDefaultDownsweepPipeline());

	ScanClass::Parameters_t push_constants;
	ScanClass::DispatchInfo_t dispatch_info;
	ScanClass::buildParameters(in_count, &push_constants, &dispatch_info, WG_SIZE);

	driver->beginScene(true);

	// Upsweeps

	for (uint32_t pass = 0; pass < dispatch_info.upsweep_pass_count; ++pass)
	{
		ScanClass::prePassParameterUpdate(pass, true, &push_constants, &dispatch_info, WG_SIZE);
		
		driver->bindComputePipeline(upsweep_pipeline.get());
		driver->bindDescriptorSets(video::EPBP_COMPUTE, upsweep_pipeline->getLayout(), 0u, 1u, &ds_upsweep.get(), nullptr);
		ScanClass::dispatchHelper(upsweep_pipeline->getLayout(), push_constants, dispatch_info, driver);

		ScanClass::postPassParameterUpdate(pass, true, &push_constants, &dispatch_info, WG_SIZE);
	}

	// Downsweeps

	for (uint32_t pass = 0; pass < dispatch_info.downsweep_pass_count; ++pass)
	{
		ScanClass::prePassParameterUpdate(pass, false, &push_constants, &dispatch_info, WG_SIZE);

		driver->bindComputePipeline(downsweep_pipeline.get());
		driver->bindDescriptorSets(video::EPBP_COMPUTE, downsweep_pipeline->getLayout(), 0u, 1u, &ds_downsweep.get(), nullptr);
		ScanClass::dispatchHelper(downsweep_pipeline->getLayout(), push_constants, dispatch_info, driver);

		ScanClass::postPassParameterUpdate(pass, false, &push_constants, &dispatch_info, WG_SIZE);
	}

	driver->endScene();

	delete[] in;

	return 0;
}
#endif

#if 0
int main()
{
	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check nbl::SIrrlichtCreationParameters
	nbl::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	params.WindowSize = dimension2d<uint32_t>(512, 512);
	params.Fullscreen = false;
	params.Vsync = true; //! If supported by target platform
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	auto device = createDeviceEx(params);

	if (!device)
		return 1; // could not create selected driver.

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
		const size_t in_count = 1 << 23; // this param is tied to macros in Histogram.comp for now
		const size_t in_size = in_count * sizeof(SortElement);

		SortElement* in_array = new SortElement[in_count];
		srand(seed++);
		for (size_t i = 0u; i < in_count; ++i)
		{
			in_array[i].key = rand();
			in_array[i].data = i;
		}
		
		auto in_array_gpu = driver->createFilledDeviceLocalGPUBufferOnDedMem(in_size, in_array);
		auto out_array_gpu = driver->createDeviceLocalGPUBufferOnDedMem(in_size);

		// Begin Radix Sort

		const int bits_per_pass = 4;
		const int buckets_count = 1 << bits_per_pass;
		const int wg_dim = 1 << 8; // limited by number of threads in the hardware for current bits_per_pass == 4
		int wg_count = (in_count + wg_dim - 1) / wg_dim;

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

		// Upsweep pipeline and ds

		smart_refctd_ptr<IGPUComputePipeline> upsweep_pipeline = nullptr;
		smart_refctd_ptr<IGPUDescriptorSet> ds_upsweep = nullptr;
		{
			const uint32_t count = 1u;
			IGPUDescriptorSetLayout::SBinding binding[count];
			for (uint32_t i = 0; i < count; ++i)
				binding[i] = { i, asset::EDT_STORAGE_BUFFER, 1u, IGPUSpecializedShader::ESS_COMPUTE, nullptr };

			auto ds_layout_gpu = driver->createGPUDescriptorSetLayout(binding, binding + count);
			ds_upsweep = driver->createGPUDescriptorSet(smart_refctd_ptr(ds_layout_gpu));

			auto pipeline_layout = driver->createGPUPipelineLayout(nullptr, nullptr, smart_refctd_ptr(ds_layout_gpu));

			smart_refctd_ptr<IGPUSpecializedShader> shader_gpu = nullptr;
			{
				auto file = smart_refctd_ptr<io::IReadFile>(filesystem->createAndOpenFile("../Upsweep.comp"));

				asset::IAssetLoader::SAssetLoadParams lp;
				auto cs_bundle = am->getAsset("../Upsweep.comp", lp);
				auto cs = smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*cs_bundle.getContents().begin());
				auto cs_rawptr = cs.get();

				shader_gpu = driver->getGPUObjectsFromAssets(&cs_rawptr, &cs_rawptr + 1)->front();
			}

			upsweep_pipeline = driver->createGPUComputePipeline(nullptr, std::move(pipeline_layout), std::move(shader_gpu));
		}

		// Downsweep pipeline and ds

		smart_refctd_ptr<IGPUComputePipeline> downsweep_pipeline = nullptr;
		smart_refctd_ptr<IGPUDescriptorSet> ds_downsweep = nullptr;
		{
			const uint32_t count = 1u;
			IGPUDescriptorSetLayout::SBinding binding[count];
			for (uint32_t i = 0; i < count; ++i)
				binding[i] = { i, asset::EDT_STORAGE_BUFFER, 1u, IGPUSpecializedShader::ESS_COMPUTE, nullptr };

			auto ds_layout_gpu = driver->createGPUDescriptorSetLayout(binding, binding + count);
			ds_downsweep = driver->createGPUDescriptorSet(smart_refctd_ptr(ds_layout_gpu));

			auto pipeline_layout = driver->createGPUPipelineLayout(nullptr, nullptr, smart_refctd_ptr(ds_layout_gpu));

			smart_refctd_ptr<IGPUSpecializedShader> shader_gpu = nullptr;
			{
				auto file = smart_refctd_ptr<io::IReadFile>(filesystem->createAndOpenFile("../Downsweep.comp"));

				asset::IAssetLoader::SAssetLoadParams lp;
				auto cs_bundle = am->getAsset("../Downsweep.comp", lp);
				auto cs = smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*cs_bundle.getContents().begin());
				auto cs_rawptr = cs.get();

				shader_gpu = driver->getGPUObjectsFromAssets(&cs_rawptr, &cs_rawptr + 1)->front();
			}

			downsweep_pipeline = driver->createGPUComputePipeline(nullptr, std::move(pipeline_layout), std::move(shader_gpu));
		}

		driver->beginScene(true);

		const uint32_t pass_count = 32 / bits_per_pass;

		for (uint32_t pass = 0; pass < pass_count; ++pass)
		{
			{
				const uint32_t count = 2;
				IGPUDescriptorSet::SDescriptorInfo ds_info[count];
				ds_info[0].desc = ((pass % 2) ? out_array_gpu : in_array_gpu);
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
			driver->dispatch(wg_count, 1, 1);
			
			video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

			uint32_t* histogram_cpu = DebugGPUBufferDownload<uint32_t>(histogram_gpu, histogram_size, driver);
			if (!histogram_cpu) __debugbreak();
			
			const uint32_t scan_wg_dim = 1 << 8;
			const uint32_t scan_in_count = histogram_count;
			// assert(scan_in_count != 1u);
			const size_t scan_in_size = histogram_size;

			// Upsweeps

			uint32_t upsweep_pass_count = std::ceil(log(scan_in_count) / log(wg_dim));

			uint32_t* cpu_scan = new uint32_t[scan_in_count];
			memcpy(cpu_scan, histogram_cpu, scan_in_size);
			
			// Initial conditions

			uint32_t stride = 1u;
			uint32_t pass_in_count = scan_in_count;

			for (uint32_t upsweep_pass = 0; upsweep_pass < upsweep_pass_count; ++upsweep_pass)
			{
				uint32_t scan_wg_count = (pass_in_count + scan_wg_dim - 1) / scan_wg_dim;
			
				{
					const uint32_t count = 1;
					IGPUDescriptorSet::SDescriptorInfo ds_info[count];
					
					ds_info[0].desc = histogram_gpu;
					ds_info[0].buffer = { 0u, histogram_size };
					
					IGPUDescriptorSet::SWriteDescriptorSet writes[count];
					for (uint32_t i = 0; i < count; ++i)
						writes[i] = { ds_upsweep.get(), i, 0u, 1u, asset::EDT_STORAGE_BUFFER, ds_info + i };
					
					driver->updateDescriptorSets(count, writes, 0u, nullptr);
				}

				driver->bindComputePipeline(upsweep_pipeline.get());
				driver->bindDescriptorSets(video::EPBP_COMPUTE, upsweep_pipeline->getLayout(), 0u, 1u, &ds_upsweep.get(), nullptr);

				driver->pushConstants(upsweep_pipeline->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(uint32_t),
					&stride);
				driver->pushConstants(upsweep_pipeline->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 4u, sizeof(uint32_t),
					&scan_in_count);
				driver->dispatch(scan_wg_count, 1, 1);

				video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

				// Check the result for this pass

				if (upsweep_pass < upsweep_pass_count - 1)
				{
					std::cout << "=========================" << std::endl;
					std::cout << "Upsweep Pass #" << upsweep_pass << std::endl;
					std::cout << "=========================" << std::endl;

					for (uint32_t wg = 0; wg < scan_wg_count; ++wg)
					{
						size_t begin = wg * scan_wg_dim;
						size_t end = (wg + 1) * scan_wg_dim;

						uint32_t* local_prefix_sum = new uint32_t[scan_wg_dim];

						uint32_t k = 0;
						uint32_t sum = 0;
						for (size_t i = begin; i < end; ++i)
						{
							size_t idx = STRIDED_IDX(i);
							sum += cpu_scan[idx];
							local_prefix_sum[k++] = sum;
						}

						k = 0;
						for (size_t i = begin; i < end; ++i)
						{
							size_t idx = STRIDED_IDX(i);
							cpu_scan[idx] = local_prefix_sum[k++];
						}

						delete[] local_prefix_sum;
					}

					DebugCompareGPUvsCPU<uint32_t>(histogram_gpu, cpu_scan, histogram_size, driver);
				}
				else
				{
					std::cout << "=========================" << std::endl;
					std::cout << "Top-of-Hierarchy Pass" << std::endl;
					std::cout << "=========================" << std::endl;

					uint32_t* local_prefix_sum = new uint32_t[pass_in_count];

					uint32_t k = 0;
					uint32_t sum = 0;
					for (size_t i = 0; i < pass_in_count; ++i)
					{
						size_t idx = STRIDED_IDX(i);
						local_prefix_sum[k++] = sum;
						sum += cpu_scan[idx];
					}

					k = 0;
					for (size_t i = 0; i < pass_in_count; ++i)
					{
						size_t idx = STRIDED_IDX(i);
						cpu_scan[idx] = local_prefix_sum[k++];
					}

					delete[] local_prefix_sum;

					DebugCompareGPUvsCPU<uint32_t>(histogram_gpu, cpu_scan, histogram_size, driver);
				}

				// Prepare for next pass

				if (upsweep_pass != upsweep_pass_count - 1u)
				{
					stride *= scan_wg_dim;
					pass_in_count = scan_wg_count;
				}
				else
				{
					stride /= scan_wg_dim;
				}
			}
			
			// Downsweep passes
			uint32_t downsweep_pass_count = upsweep_pass_count - 1;
			
			for (uint32_t downsweep_pass = 0; downsweep_pass < downsweep_pass_count; ++downsweep_pass)
			{
				uint32_t pass_out_count = pass_in_count * scan_wg_dim;
				uint32_t scan_wg_count = (pass_out_count + scan_wg_dim - 1) / scan_wg_dim;
				
				{
					const uint32_t count = 1;
					IGPUDescriptorSet::SDescriptorInfo ds_info[count];
			
					ds_info[0].desc = histogram_gpu;
					ds_info[0].buffer = { 0u, histogram_size };
			
					IGPUDescriptorSet::SWriteDescriptorSet writes[count];
					for (uint32_t i = 0; i < count; ++i)
						writes[i] = { ds_downsweep.get(), i, 0u, 1u, asset::EDT_STORAGE_BUFFER, ds_info + i };
			
					driver->updateDescriptorSets(count, writes, 0u, nullptr);
				}
			
				driver->bindComputePipeline(downsweep_pipeline.get());
				driver->bindDescriptorSets(video::EPBP_COMPUTE, downsweep_pipeline->getLayout(), 0u, 1u, &ds_downsweep.get(), nullptr);
			
				driver->pushConstants(downsweep_pipeline->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(uint32_t),
					&stride);
				driver->dispatch(scan_wg_count, 1, 1);
			
				video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
			
				// Check the result for this pass
			
				std::cout << "=========================" << std::endl;
				std::cout << "Downsweep Pass #" << downsweep_pass << std::endl;
				std::cout << "=========================" << std::endl;
			
				
				for (uint32_t wg = 0; wg < scan_wg_count; ++wg)
				{
					size_t begin = wg * scan_wg_dim;
					size_t end = (wg + 1) * scan_wg_dim;
			
					uint32_t* downsweep_result = new uint32_t[scan_wg_dim];
			
					size_t idx = STRIDED_IDX(end - 1);
					downsweep_result[0] = cpu_scan[idx];
			
					uint32_t k = 1;
					for (size_t i = begin + 1; i < end; ++i)
						downsweep_result[k++] = cpu_scan[STRIDED_IDX(i - 1)] + downsweep_result[0];
			
					k = 0;
					for (size_t i = begin; i < end; ++i)
					{
						size_t idx = STRIDED_IDX(i);
						cpu_scan[idx] = downsweep_result[k++];
					}
			
					delete[] downsweep_result;
				}
			
				DebugCompareGPUvsCPU<uint32_t>(histogram_gpu, cpu_scan, histogram_size, driver);
			
				// Prepare for the next pass
			
				pass_in_count = pass_out_count;
				stride /= scan_wg_dim;
			}

			delete[] cpu_scan;
			
			{
				const uint32_t count = 3;
				IGPUDescriptorSet::SDescriptorInfo ds_info[count];
				ds_info[0].desc = ((pass % 2) ? out_array_gpu : in_array_gpu);
				ds_info[0].buffer = { 0u, in_size };
			
				ds_info[1].desc = ((pass % 2) ? in_array_gpu : out_array_gpu);
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
			driver->dispatch(wg_count, 1, 1);
			
			video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		}

		// Testing

		// Validate local histograms and sort
		
		// uint32_t* dataFromBuffer = DebugGPUBufferDownload<uint32_t>(histogram_gpu, histogram_size, driver);
		// SortElement* dataFromBuffer = DebugGPUBufferDownload<SortElement>(out_array_gpu, in_size, driver);

		// if (dataFromBuffer)
		// {
		// 	uint32_t* histogram_cpu = new uint32_t[histogram_count];
		// 	for (size_t wg = 0; wg < wg_count; ++wg)
		// 	{
		// 		size_t begin = wg * wg_dim;
		// 		size_t end = (wg + 1) * wg_dim;
		// 	
		// 		// std::stable_sort(in_array + begin, in_array + end, [](const SortElement& a, const SortElement& b) { return (a.key & 0xf) < (b.key & 0xf); });
				// 
				// for (uint32_t i = begin; i < end; ++i)
				// {
				// 	if ((dataFromBuffer[i].key != in_array[i].key) || (dataFromBuffer[i].data != in_array[i].data))
				// 		__debugbreak();
				// }
		// 	
		// 		uint32_t local_histogram[16] = { 0 };
		// 		
		// 		for (size_t i = begin; i < end; ++i)
		// 		{
		// 			++local_histogram[in_array[i].key & 0xf];
		// 		}
		// 		
		// 		for (uint32_t i = 0; i < 16; ++i)
		// 		{
		// 			histogram_cpu[i * wg_count + wg] = local_histogram[i];
		// 		}
		// 	}
		// 
		// 	uint32_t* ps_histogram_cpu = new uint32_t[histogram_count];
		// 	
		// 	uint32_t sum = 0;
		// 	for (size_t i = 0; i < histogram_count; ++i)
		// 	{
		// 		ps_histogram_cpu[i] = sum;
		// 		sum += histogram_cpu[i];
		// 	}
		// 
		// 	// ps_histogram_cpu should be available to index from
		// 
		// 
		// 	// for (size_t wg = 0; wg < wg_count; ++wg)
		// 	// {
		// 	// 	size_t begin = wg * wg_dim;
		// 	// 	size_t end = (wg + 1) * wg_dim;
		// 	// 
		// 	// 	std::stable_sort(in_array + begin, in_array + end, [](const SortElement& a, const SortElement& b) { return (a.key & 0xf) < (b.key & 0xf); });
		// 	// 
		// 	// 	uint32_t local_histogram[16] = { 0 };
		// 	// 
		// 	// 	// Populate the local_histogram first with relevant values and __then__ compare
		// 	// 	for (int i = 0; i < buckets_count; ++i)
		// 	// 		local_histogram[i] = ps_histogram_cpu[i * wg_count + wg];
		// 	// 
		// 	// 	for (size_t i = begin; i < end; ++i)
		// 	// 	{
		// 	// 		uint32_t digit = (in_array[i].key & 0xf);
		// 	// 		if (local_histogram[digit] != dataFromBuffer[i].key)
		// 	// 			__debugbreak();
		// 	// 	}
		// 	// }
		// 
		// 	// for (size_t i = 0; i < 16 * wg_count; ++i)
		// 	// {
		// 	// 	if (histogram_cpu[i] != dataFromBuffer[i])
		// 	// 		__debugbreak();
		// 	// }
		// 	// 
		// 	// for (size_t i = 0; i < in_count; ++i)
		// 	// {
		// 	// 	if (dataFromBuffer[i] != ??)
		// 	// }
		// 
		// 	std::cout << "PASS" << std::endl;
		// 
		// 	delete[] histogram_cpu;
		// }

		std::cout << "CPU Sort begins" << std::endl;
		std::stable_sort(in_array, in_array + in_count, [](const SortElement& a, const SortElement& b) { return a.key < b.key; });
		std::cout << "CPU Sort ends" << std::endl;

		DebugCompareGPUvsCPU<SortElement>(in_array_gpu, in_array, in_size, driver);

		driver->endScene();

		delete[] in_array;
#if 1
	}
#endif

	return 0;
}
#endif