#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../../source/Nabla/COpenGLDriver.h"

using namespace nbl;
using namespace core;
using namespace video;

struct SortElement { uint32_t key, data; };

// Todo: Should I pass smart_refctd_ptr by const ref/ref?

// Todo: There's a lot of room for optimization here, especially creating all the gpu resources again
// and again when they can be created once and reused

// Works for every POT in [2^0, 2^24], should even work for higher powers but for some
// reason I can't download the array to the CPU with the current code, to check

void ExclusiveSumScanGPU(smart_refctd_ptr<IGPUBuffer> in, const size_t in_count, core::smart_refctd_ptr<IrrlichtDevice> device)
{
	assert(in_count != 0);

	IVideoDriver* driver = device->getVideoDriver();
	io::IFileSystem* filesystem = device->getFileSystem();
	asset::IAssetManager* am = device->getAssetManager();

	const uint32_t wg_dim = 1 << 10;
	const uint32_t wg_count = (in_count + wg_dim - 1) / wg_dim;
	const size_t in_size = in_count * sizeof(uint32_t);

	if (wg_count == 1)
	{
		smart_refctd_ptr<IGPUDescriptorSet> ds_local_prefix_sum = nullptr;
		smart_refctd_ptr<IGPUComputePipeline> local_prefix_sum_pipeline = nullptr;
		{
			const uint32_t count = 1u;
			IGPUDescriptorSetLayout::SBinding binding[count];
			for (uint32_t i = 0; i < count; ++i)
				binding[i] = { i, asset::EDT_STORAGE_BUFFER, 1u, IGPUSpecializedShader::ESS_COMPUTE, nullptr };

			auto ds_layout_gpu = driver->createGPUDescriptorSetLayout(binding, binding + count);
			ds_local_prefix_sum = driver->createGPUDescriptorSet(smart_refctd_ptr(ds_layout_gpu));

			auto pipeline_layout = driver->createGPUPipelineLayout(nullptr, nullptr, smart_refctd_ptr(ds_layout_gpu));

			smart_refctd_ptr<IGPUSpecializedShader> shader_gpu = nullptr;
			{
				auto file = smart_refctd_ptr<io::IReadFile>(filesystem->createAndOpenFile("../PrefixSumLocal.comp"));

				asset::IAssetLoader::SAssetLoadParams lp;
				auto cs_bundle = am->getAsset("../PrefixSumLocal.comp", lp);
				auto cs = smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*cs_bundle.getContents().begin());
				auto cs_rawptr = cs.get();

				shader_gpu = driver->getGPUObjectsFromAssets(&cs_rawptr, &cs_rawptr + 1)->front();
			}

			local_prefix_sum_pipeline = driver->createGPUComputePipeline(nullptr, std::move(pipeline_layout), std::move(shader_gpu));
		}

		{
			const uint32_t count = 1;
			IGPUDescriptorSet::SDescriptorInfo ds_info[count];

			// Todo: You probably don't need it to update every iteration
			ds_info[0].desc = in;
			ds_info[0].buffer = { 0u, in_size };

			IGPUDescriptorSet::SWriteDescriptorSet writes[count];
			for (uint32_t i = 0; i < count; ++i)
				writes[i] = { ds_local_prefix_sum.get(), i, 0u, 1u, asset::EDT_STORAGE_BUFFER, ds_info + i };

			driver->updateDescriptorSets(count, writes, 0u, nullptr);

			driver->bindComputePipeline(local_prefix_sum_pipeline.get());
			driver->bindDescriptorSets(video::EPBP_COMPUTE, local_prefix_sum_pipeline->getLayout(), 0u, 1u, &ds_local_prefix_sum.get(), nullptr);

			driver->pushConstants(local_prefix_sum_pipeline->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(uint32_t), &in_count);
			driver->dispatch(wg_count, 1, 1);	// Todo: This could be faster if you don't launch 1024 threads for every input length

			// Todo: Do I need GL_BUFFER_UPDATE_BARRIER_BIT ?
			video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
		}
	}
	else
	{	
		//! Todo: Based on the number of elements in histogram figure out how big you want this partial reductions
		//! array to be
		
		const size_t partial_reductions_count = wg_count;
		const size_t partial_reductions_size = wg_count * sizeof(uint32_t);
		auto partial_reductions_gpu = driver->createDeviceLocalGPUBufferOnDedMem(partial_reductions_size);
		
		smart_refctd_ptr<IGPUComputePipeline> partial_reductions_pipeline = nullptr;
		smart_refctd_ptr<IGPUDescriptorSet> ds_partial_reductions = nullptr;
		{
			const uint32_t count = 2u;
			IGPUDescriptorSetLayout::SBinding binding[count];
			for (uint32_t i = 0; i < count; ++i)
				binding[i] = { i, asset::EDT_STORAGE_BUFFER, 1u, IGPUSpecializedShader::ESS_COMPUTE, nullptr };

			auto ds_layout_gpu = driver->createGPUDescriptorSetLayout(binding, binding + count);
			ds_partial_reductions = driver->createGPUDescriptorSet(smart_refctd_ptr(ds_layout_gpu));

			auto pipeline_layout = driver->createGPUPipelineLayout(nullptr, nullptr, smart_refctd_ptr(ds_layout_gpu));

			smart_refctd_ptr<IGPUSpecializedShader> shader_gpu = nullptr;
			{
				auto file = smart_refctd_ptr<io::IReadFile>(filesystem->createAndOpenFile("../PrefixSumPartialReductions.comp"));

				asset::IAssetLoader::SAssetLoadParams lp;
				auto cs_bundle = am->getAsset("../PrefixSumPartialReductions.comp", lp);
				auto cs = smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*cs_bundle.getContents().begin());
				auto cs_rawptr = cs.get();

				shader_gpu = driver->getGPUObjectsFromAssets(&cs_rawptr, &cs_rawptr + 1)->front();
			}

			partial_reductions_pipeline = driver->createGPUComputePipeline(nullptr, std::move(pipeline_layout), std::move(shader_gpu));
		}

		// Dispatch for computing partial reductions

		{
			const uint32_t count = 2;
			IGPUDescriptorSet::SDescriptorInfo ds_info[count];
			ds_info[0].desc = in;
			ds_info[0].buffer = { 0u, in_size };

			// Todo: You probably don't need it to update every iteration
			ds_info[1].desc = partial_reductions_gpu;
			ds_info[1].buffer = { 0u, partial_reductions_size };

			IGPUDescriptorSet::SWriteDescriptorSet writes[count];
			for (uint32_t i = 0; i < count; ++i)
				writes[i] = { ds_partial_reductions.get(), i, 0u, 1u, asset::EDT_STORAGE_BUFFER, ds_info + i };

			driver->updateDescriptorSets(count, writes, 0u, nullptr);

			driver->bindComputePipeline(partial_reductions_pipeline.get());
			driver->bindDescriptorSets(video::EPBP_COMPUTE, partial_reductions_pipeline->getLayout(), 0u, 1u, &ds_partial_reductions.get(), nullptr);

			driver->pushConstants(partial_reductions_pipeline->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(uint32_t), &in_count);
			driver->dispatch(wg_count, 1, 1);
		}

		ExclusiveSumScanGPU(partial_reductions_gpu, partial_reductions_count, device);

		// Note: PrefixSumLocal.comp and PrefixSumGlobal.comp are extremely similar. I tried to use the same shader
		// source for both, differentiating them based on the number of workgroups called, but it seems like the unused
		// interface block (for increments array) in case of the local prefix sum is causing some errors with the glsl
		// compiler. Maybe I can separate them at shader compile time?

		smart_refctd_ptr<IGPUComputePipeline> global_prefix_sum_pipeline = nullptr;
		smart_refctd_ptr<IGPUDescriptorSet> ds_global_prefix_sum = nullptr;
		{
			const uint32_t count = 2u;
			IGPUDescriptorSetLayout::SBinding binding[count];
			for (uint32_t i = 0; i < count; ++i)
				binding[i] = { i, asset::EDT_STORAGE_BUFFER, 1u, IGPUSpecializedShader::ESS_COMPUTE, nullptr };
		
			auto ds_layout_gpu = driver->createGPUDescriptorSetLayout(binding, binding + count);
			ds_global_prefix_sum = driver->createGPUDescriptorSet(smart_refctd_ptr(ds_layout_gpu));
		
			auto pipeline_layout = driver->createGPUPipelineLayout(nullptr, nullptr, smart_refctd_ptr(ds_layout_gpu));
		
			smart_refctd_ptr<IGPUSpecializedShader> shader_gpu = nullptr;
			{
				auto file = smart_refctd_ptr<io::IReadFile>(filesystem->createAndOpenFile("../PrefixSumGlobal.comp"));
		
				asset::IAssetLoader::SAssetLoadParams lp;
				auto cs_bundle = am->getAsset("../PrefixSumGlobal.comp", lp);
				auto cs = smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*cs_bundle.getContents().begin());
				auto cs_rawptr = cs.get();
		
				shader_gpu = driver->getGPUObjectsFromAssets(&cs_rawptr, &cs_rawptr + 1)->front();
			}
		
			global_prefix_sum_pipeline = driver->createGPUComputePipeline(nullptr, std::move(pipeline_layout), std::move(shader_gpu));
		}

		// Dispatch for the final global scan array

		{
			const uint32_t count = 2;
			IGPUDescriptorSet::SDescriptorInfo ds_info[count];
			ds_info[0].desc = in;
			ds_info[0].buffer = { 0u, in_size };

			// Todo: You probably don't need it to update every iteration
			ds_info[1].desc = partial_reductions_gpu;
			ds_info[1].buffer = { 0u, partial_reductions_size };

			IGPUDescriptorSet::SWriteDescriptorSet writes[count];
			for (uint32_t i = 0; i < count; ++i)
				writes[i] = { ds_global_prefix_sum.get(), i, 0u, 1u, asset::EDT_STORAGE_BUFFER, ds_info + i };

			driver->updateDescriptorSets(count, writes, 0u, nullptr);

			driver->bindComputePipeline(global_prefix_sum_pipeline.get());
			driver->bindDescriptorSets(video::EPBP_COMPUTE, global_prefix_sum_pipeline->getLayout(), 0u, 1u, &ds_global_prefix_sum.get(), nullptr);

			driver->pushConstants(global_prefix_sum_pipeline->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(uint32_t), &in_count);
			driver->dispatch(wg_count, 1, 1);
		}
	}
}

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

	GLint64 max_ssbo_size;
	video::COpenGLExtensionHandler::extGlGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &max_ssbo_size);

	std::cout << "\nMax SSBO size: " << max_ssbo_size << " bytes" << std::endl;

	GLint max_wg_count;
	video::COpenGLExtensionHandler::extGlGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &max_wg_count);

	std::cout << "Max WG count: " << max_wg_count << std::endl;

	GLint max_wg_dim;
	video::COpenGLExtensionHandler::extGlGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &max_wg_dim);
	std::cout << "Max WG dim: " << max_wg_dim << std::endl;

	const size_t in_count = 1 << 28; // > 64 MB, this param is tied to macros in Histogram.comp for now
	const size_t in_size = in_count * sizeof(SortElement);
	
	SortElement* in_array = new SortElement[in_count];
	srand(666);
	for (size_t i = 0u; i < in_count; ++i)
	{
		in_array[i].key = (rand() % 16);
		in_array[i].data = rand();
	}
	
	auto in_array_gpu = driver->createFilledDeviceLocalGPUBufferOnDedMem(in_size, in_array);
	auto out_array_gpu = driver->createDeviceLocalGPUBufferOnDedMem(in_size);

	// Begin Radix Sort

	const int bits_per_pass = 4;
	const int buckets_count = 1 << bits_per_pass;
	const int values_per_thread = 4;
	const int workgroup_dim = 1 << 9; // limited by shared memory
	const int workgroup_count = in_count / (workgroup_dim * values_per_thread);

	const size_t histogram_count = workgroup_count * buckets_count;
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

	smart_refctd_ptr<IGPUComputePipeline> sort_and_scatter_pipeline = nullptr;
	smart_refctd_ptr<IGPUDescriptorSet> ds_sort_and_scatter = nullptr;
	{
		const uint32_t count = 3u;
		IGPUDescriptorSetLayout::SBinding binding[count];
		for (uint32_t i = 0; i < count; ++i)
			binding[i] = { i, asset::EDT_STORAGE_BUFFER, 1u, IGPUSpecializedShader::ESS_COMPUTE, nullptr };

		auto ds_layout_gpu = driver->createGPUDescriptorSetLayout(binding, binding + count);
		ds_sort_and_scatter = driver->createGPUDescriptorSet(smart_refctd_ptr(ds_layout_gpu));

		auto pipeline_layout = driver->createGPUPipelineLayout(nullptr, nullptr, smart_refctd_ptr(ds_layout_gpu));

		smart_refctd_ptr<IGPUSpecializedShader> shader_gpu = nullptr;
		{
			auto file = smart_refctd_ptr<io::IReadFile>(filesystem->createAndOpenFile("../SortAndScatter.comp"));

			asset::IAssetLoader::SAssetLoadParams lp;
			auto cs_bundle = am->getAsset("../SortAndScatter.comp", lp);
			auto cs = smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*cs_bundle.getContents().begin());
			auto cs_rawptr = cs.get();

			shader_gpu = driver->getGPUObjectsFromAssets(&cs_rawptr, &cs_rawptr + 1)->front();
		}

		sort_and_scatter_pipeline = driver->createGPUComputePipeline(nullptr, std::move(pipeline_layout), std::move(shader_gpu));
	}

	driver->beginScene(true);

	const uint32_t pass_count = 1u;

	for (uint32_t pass = 0; pass < pass_count; ++pass)
	{
		{
			const uint32_t count = 2;
			IGPUDescriptorSet::SDescriptorInfo ds_info[count];
			ds_info[0].desc = (pass % 2) ? histogram_gpu : in_array_gpu;
			ds_info[0].buffer = { 0u, in_size };

			ds_info[1].desc = (pass % 2) ? in_array_gpu : histogram_gpu;
			ds_info[1].buffer = { 0u, histogram_size };

			IGPUDescriptorSet::SWriteDescriptorSet writes[count];
			for (uint32_t i = 0; i < count; ++i)
				writes[i] = { ds_histogram.get(), i, 0u, 1u, asset::EDT_STORAGE_BUFFER, ds_info + i };

			driver->updateDescriptorSets(count, writes, 0u, nullptr);
		}

		driver->bindComputePipeline(histogram_pipeline.get());
		driver->bindDescriptorSets(video::EPBP_COMPUTE, histogram_pipeline->getLayout(), 0u, 1u, &ds_histogram.get(), nullptr);

		driver->pushConstants(histogram_pipeline->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(uint32_t), &pass);
		driver->dispatch(workgroup_count, 1, 1);

		// Todo: Do I need this barrier?
		video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

		ExclusiveSumScanGPU(histogram_gpu, histogram_count, device);

		// {
		// 	const uint32_t count = 3;
		// 	IGPUDescriptorSet::SDescriptorInfo ds_info[count];
		// 	ds_info[0].desc = in_array_gpu;
		// 	ds_info[0].buffer = { 0u, in_size };
		// 
		// 	ds_info[1].desc = out_array_gpu;
		// 	ds_info[1].buffer = { 0u, in_size };
		// 
		// 	ds_info[2].desc = histogram_gpu;
		// 	ds_info[2].buffer = { 0u, histogram_size };
		// 
		// 	IGPUDescriptorSet::SWriteDescriptorSet writes[count];
		// 	for (uint32_t i = 0; i < count; ++i)
		// 		writes[i] = { ds_sort_and_scatter.get(), i, 0u, 1u, asset::EDT_STORAGE_BUFFER, ds_info + i };
		// 
		// 	driver->updateDescriptorSets(count, writes, 0u, nullptr);
		// }
		// 
		// driver->bindComputePipeline(sort_and_scatter_pipeline.get());
		// driver->bindDescriptorSets(video::EPBP_COMPUTE, sort_and_scatter_pipeline->getLayout(), 0u, 1u, &ds_sort_and_scatter.get(), nullptr);
		// 
		// driver->dispatch(in_count/workgroup_dim, 1, 1);
		
		// std::cout << "Pass #" << pass << std::endl;
		// std::cout << "---------------------------" << std::endl;

		constexpr uint64_t timeout_ns = 15000000000u;
		const uint32_t alignment = sizeof(SortElement); // sizeof(uint32_t);
		auto downloadStagingArea = driver->getDefaultDownStreamingBuffer();
		auto downBuffer = downloadStagingArea->getBuffer();
		
		bool success = false;
		
		uint32_t array_size_32 = uint32_t(histogram_size);
		uint32_t address = std::remove_pointer<decltype(downloadStagingArea)>::type::invalid_address;
		auto unallocatedSize = downloadStagingArea->multi_alloc(1u, &address, &array_size_32, &alignment);
		if (unallocatedSize)
		{
			os::Printer::log("Could not download the buffer from the GPU!", ELL_ERROR);
			return false;
		}

		// (pass % 2) ? driver->copyBuffer(in_array_gpu.get(), downBuffer, 0, address, array_size_32)
		// 	: driver->copyBuffer(histogram_gpu.get(), downBuffer, 0, address, array_size_32);

		// driver->copyBuffer(out_array_gpu.get(), downBuffer, 0, address, array_size_32);
		driver->copyBuffer(histogram_gpu.get(), downBuffer, 0, address, array_size_32);
		
		auto downloadFence = driver->placeFence(true);
		auto result = downloadFence->waitCPU(timeout_ns, true);
		
		uint32_t* dataFromBuffer = nullptr;
		// SortElement* dataFromBuffer = nullptr;
		if (result != video::E_DRIVER_FENCE_RETVAL::EDFR_TIMEOUT_EXPIRED && result != video::E_DRIVER_FENCE_RETVAL::EDFR_FAIL)
		{
			if (downloadStagingArea->needsManualFlushOrInvalidate())
				driver->invalidateMappedMemoryRanges({ {downloadStagingArea->getBuffer()->getBoundMemory(),address,array_size_32} });
		
			dataFromBuffer = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(downloadStagingArea->getBufferPointer()) + address);
			// dataFromBuffer = reinterpret_cast<SortElement*>(reinterpret_cast<uint8_t*>(downloadStagingArea->getBufferPointer()) + address);
		}
		else
		{
			os::Printer::log("Could not download the buffer from the GPU, fence not signalled!", ELL_ERROR);
		}
		
		downloadStagingArea->multi_free(1u, &address, &array_size_32, nullptr);
		
		if (dataFromBuffer)
		{
			bool success = true;

			// Generate histogram

			uint32_t* histogram = new uint32_t[histogram_count];
			memset(histogram, 0, histogram_size);
			
			const uint32_t values_per_group = values_per_thread * workgroup_dim;
			for (uint32_t wg = 0; wg < workgroup_count; ++wg)
			{
				for (uint32_t i = wg * values_per_group; i < (wg + 1) * values_per_group; ++i)
					++histogram[wg * buckets_count + in_array[i].key];
			}

			// Remap histogram

			uint32_t* histogram_remap = new uint32_t[histogram_count];
			memset(histogram_remap, 0, histogram_size);

			{
				int k = 0;
				for (uint32_t i = 0; i < buckets_count; ++i)
				{
					for (uint32_t wg = 0; wg < workgroup_count; ++wg)
					{
						histogram_remap[k++] = histogram[i + wg * buckets_count];
					}
				}
			}

			// Now do the actual prefix sum

			uint32_t* prefix_sum_scan = new uint32_t[histogram_count];
			memset(prefix_sum_scan, 0, histogram_size);
			{
				uint32_t sum = 0;
				for (uint32_t i = 0; i < histogram_count; ++i)
				{
					prefix_sum_scan[i] = sum;
					sum += histogram_remap[i];
				}
			}

			// Validate sort and scatter
			// for (uint32_t i = 0; i < in_count; ++i)
			// {
			// 	if ((dataFromBuffer[i].key != in_array[i].key) || (dataFromBuffer[i].data != in_array[i].data))
			// 	{
			// 		__debugbreak();
			// 	}
			// }

			// Validate prefix sum scan of histogram
			int k = 0;
			for (uint32_t i = 0; i < histogram_count; ++i)
			{
				if (prefix_sum_scan[i] != dataFromBuffer[i])
				{
					success = false;
					__debugbreak();
					break;
				}
			}
			
			// Validate histogram
			// int k = 0;
			// for (uint32_t i = 0; i < buckets_count; ++i)
			// {
			// 	for (uint32_t wg = 0; wg < workgroup_count; ++wg)
			// 	{
			// 		if (dataFromBuffer[k++] != histogram[i + wg * buckets_count])
			// 		{
			// 			success = false;
			// 			__debugbreak();
			// 			break;
			// 		}
			// 	}
			// }
			
			delete[] histogram;
			delete[] histogram_remap;
			delete[] prefix_sum_scan;

			// const size_t wg_dim = 1 << 10;
			// const size_t wg_count = in_count / wg_dim;
			// uint32_t* partial_red_cpu = new uint32_t[wg_count];
			// for (uint32_t wg = 0; wg < wg_count; ++wg)
			// {
			// 	uint32_t sum = 0;
			// 	for (uint32_t i = 0; i < wg_dim; ++i)
			// 	{
			// 		// prefix_sum_scan[workgroup_dim * wg + i] = sum;
			// 		sum += in_array[wg_dim * wg + i];
			// 	}
			// 
			// 	partial_red_cpu[wg] = sum;
			// }

			// uint32_t* increments = new uint32_t[workgroup_dim];
			// {
			// 	uint32_t sum = 0;
			// 	for (uint32_t i = 0; i < workgroup_dim; ++i)
			// 	{
			// 		increments[i] = sum;
			// 		sum += partial_red_cpu[i];
			// 	}
			// }

			// delete[] partial_red_cpu;
			// delete[] increments;

			std::cout << "---------------------------\n" << std::endl;

			std::cout << "PASS" << std::endl;
		}
	}
	
	driver->endScene();

	delete[] in_array;

	return 0;
}