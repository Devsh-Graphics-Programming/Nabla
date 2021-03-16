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

	unsigned int seed = 0;

#define STRESS_TEST 0

#if STRESS_TEST
	while (true)
	{
#endif

		// I think you cannot download more than 64 MB of data currently though
		const size_t in_count = 1 << 23; // > 64 MB, this param is tied to macros in Histogram.comp for now
		const size_t in_size = in_count * sizeof(SortElement);
		// const size_t in_size = in_count * sizeof(uint32_t);

		SortElement* in_array = new SortElement[in_count];
		// uint32_t* in_array = new uint32_t[in_count];
		srand(seed++);
		for (size_t i = 0u; i < in_count; ++i)
		{
			// in_array[i] = rand();
			in_array[i].key = (rand() % 16);
			in_array[i].data = i;
		}


		

		auto in_array_gpu = driver->createFilledDeviceLocalGPUBufferOnDedMem(in_size, in_array);
		auto out_array_gpu = driver->createDeviceLocalGPUBufferOnDedMem(in_size);

		// Begin Radix Sort

		const int bits_per_pass = 4;
		const int buckets_count = 1 << bits_per_pass;
		const int wg_dim = 1 << 8; // limited by number of threads in the hardware for current bits_per_pass == 4
		const int wg_count = in_count / wg_dim;

		const size_t histogram_count = wg_count * buckets_count;
		const size_t histogram_size = histogram_count * sizeof(uint32_t);
		auto histogram_gpu = driver->createDeviceLocalGPUBufferOnDedMem(histogram_size);

		// uint32_t* ping = new uint32_t[in_count];
		// memcpy(ping, in_array, in_size);
		// 
		// uint32_t* pong = new uint32_t[in_count];
		// 
		// for (int pass = 0; pass < 4; ++pass)
		// {
		// 	uint32_t shift = 4 * pass;
		// 	uint32_t* in = pass ? pong : ping;
		// 	uint32_t* out = pass ? ping : pong;
		// 
		// 	uint32_t* bit_shifted_in = new uint32_t[in_count];
		// 	for (int i = 0; i < in_count; ++i)
		// 		bit_shifted_in[i] = ((in[i] >> shift) & 0xf);
		// 
		// 	uint32_t* histogram = new uint32_t[histogram_count];
		// 	memset(histogram, 0, histogram_size);
		// 
		// 	for (uint32_t wg = 0; wg < wg_count; ++wg)
		// 	{
		// 		for (uint32_t i = wg * wg_dim; i < (wg + 1) * wg_dim; ++i)
		// 			++histogram[wg * buckets_count + bit_shifted_in[i]];
		// 	}
		// 
		// 	uint32_t* ps_histogram = new uint32_t[histogram_count];
		// 	uint32_t sum = 0;
		// 	for (int i = 0; i < histogram_count; ++i)
		// 	{
		// 		ps_histogram[i] = sum;
		// 		sum += histogram[i];
		// 	}
		// 
		// 	for (int i = 0; i < in_count; ++i)
		// 	{
		// 		uint32_t key = in[i];
		// 		uint32_t digit = (key >> shift) & 0xf;
		// 
		// 		uint32_t local_offset = ps_histogram[digit];
		// 		out[local_offset] = key;
		// 		ps_histogram[digit]++;
		// 	}
		// 
		// 	// std::cout << "Output of pass 0" << std::endl;
		// 	// for (uint32_t i = 0; i < in_count; ++i)
		// 	// {
		// 	// 	std::cout << out[i] << ",\t";
		// 	// 	if (((i + 1) % 32) == 0) std::cout << std::endl;
		// 	// }
		// 	// std::cout << std::endl;
		// 
		// 	delete[] ps_histogram;
		// 	delete[] bit_shifted_in;
		// 	delete[] histogram;
		// }
		// 
		// for (int i = 0; i < in_count; ++i)
		// {
		// 	if (pong[i] != in_array[i]) // Todo: compare with output of pass 0
		// 		__debugbreak();
		// 	// std::cout << i << ":\t" << pong[i] << std::endl;
		// }
		// 
		// std::cout << "PONG after pass 3 is same as the output of pass 0" << std::endl;
		// 
		// std::cout << std::endl;

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

		const uint32_t pass_count = 32 / bits_per_pass;

		for (uint32_t pass = 0; pass < 1; ++pass)
		{
			{
				const uint32_t count = 2;
				IGPUDescriptorSet::SDescriptorInfo ds_info[count];
				ds_info[0].desc = (pass ? out_array_gpu : in_array_gpu);
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

			// Todo: Do I need GL_BUFFER_UPDATE_BARRIER_BIT? Do I need this barrier at all?
			video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

			ExclusiveSumScanGPU(histogram_gpu, histogram_count, device);

				// Todo: Do I need GL_BUFFER_UPDATE_BARRIER_BIT? Do I need this barrier at all?
				video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

				{
					const uint32_t count = 3;
					IGPUDescriptorSet::SDescriptorInfo ds_info[count];
					ds_info[0].desc = (pass ? out_array_gpu : in_array_gpu);
					ds_info[0].buffer = { 0u, in_size };

					ds_info[1].desc = (pass ? in_array_gpu : out_array_gpu);
					ds_info[1].buffer = { 0u, in_size };

					ds_info[2].desc = histogram_gpu;
					ds_info[2].buffer = { 0u, histogram_size };

					IGPUDescriptorSet::SWriteDescriptorSet writes[count];
					for (uint32_t i = 0; i < count; ++i)
						writes[i] = { ds_sort_and_scatter.get(), i, 0u, 1u, asset::EDT_STORAGE_BUFFER, ds_info + i };

					driver->updateDescriptorSets(count, writes, 0u, nullptr);
				}

				driver->bindComputePipeline(sort_and_scatter_pipeline.get());
				driver->bindDescriptorSets(video::EPBP_COMPUTE, sort_and_scatter_pipeline->getLayout(), 0u, 1u, &ds_sort_and_scatter.get(),
					nullptr);

				driver->pushConstants(sort_and_scatter_pipeline->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(uint32_t),
					&shift);
				driver->dispatch(wg_count, 1, 1);

				// Todo: Do I need GL_BUFFER_UPDATE_BARRIER_BIT? Do I need this barrier at all?
				video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
			if (pass == 0)
			{
			}
			
			// std::cout << "=================" << std::endl;
			// std::cout << "Pass #" << pass << std::endl;
			// std::cout << "=================" << std::endl;
			// 
			// uint32_t* dataFromBuffer = DebugGPUBufferDownload<uint32_t>((pass ? in_array_gpu : out_array_gpu), in_size, driver);
			// 
			// for (uint32_t i = 0; i < in_count; ++i)
			// 	std::cout << i << "\t" << dataFromBuffer[i] << std::endl;
			// 
			// std::cout << std::endl;
		}

#if 1
		// Testing
		
		// uint32_t* dataFromBuffer = DebugGPUBufferDownload<uint32_t>(in_array_gpu, in_size, driver);
		SortElement* dataFromBuffer = DebugGPUBufferDownload<SortElement>(out_array_gpu, in_size, driver);

		uint32_t* global_histogram = new uint32_t[buckets_count];
		memset(global_histogram, 0, buckets_count * sizeof(uint32_t));
		for (uint32_t i = 0; i < in_count; ++i)
			++global_histogram[in_array[i].key];

		int k = 0;
		for (int i = 0; i < buckets_count; ++i)
		{
			uint32_t prev_data = dataFromBuffer[k++].data;
			for (int j = 1; j < global_histogram[i]; ++j)
			{
				if (dataFromBuffer[k].data < prev_data)
					__debugbreak();

				prev_data = dataFromBuffer[k++].data;
			}
		}

		std::cout << "PASS" << std::endl;

		// if (dataFromBuffer)
		// {
		// 	for (int i = 0; i < in_count; ++i)
		// 	{
		// 		std::cout << "(" << dataFromBuffer[i].key << ", \t" << dataFromBuffer[i].data << ")" << std::endl;
		// 	}
		// 	
		// 	// std::cout << "PASS" << std::endl;
		// }


#if 0

				// Remap histogram

				uint32_t* histogram_remap = new uint32_t[histogram_count];
				memset(histogram_remap, 0, histogram_size);

				{
					int k = 0;
					for (uint32_t i = 0; i < buckets_count; ++i)
					{
						for (uint32_t wg = 0; wg < wg_count; ++wg)
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

				for (int wg = 0; wg < wg_count; ++wg)
				{
					size_t begin = wg * wg_dim;
					size_t end = (wg + 1) * wg_dim;
					std::sort(in_array + begin, in_array + end);

					for (int i = begin; i < end; ++i)
					{
						int digit = in_array[i];
						int global_offset = prefix_sum_scan[digit * wg_count + wg];
						if (global_offset != dataFromBuffer[i])
						{
							__debugbreak();
						}
					}
				}

				// Validate prefix sum scan of histogram

				// int k = 0;
				// for (uint32_t i = 0; i < histogram_count; ++i)
				// {
				// 	if (prefix_sum_scan[i] != dataFromBuffer[i])
				// 	{
				// 		__debugbreak();
				// 		break;
				// 	}
				// }

				// Todo: remap again --for multiple wgs

				delete[] histogram_remap;
				delete[] prefix_sum_scan;
#endif
#if 0
				for (int wg = 0; wg < wg_count; ++wg)
				{
					size_t begin = wg * wg_dim;
					size_t end = (wg + 1) * wg_dim;
				
					// Generate a local histogram and take prefix sum scan of that to index from
				
					uint32_t* local_histogram = new uint32_t[buckets_count];
					memset(local_histogram, 0, buckets_count * sizeof(uint32_t));
				
					for (size_t i = begin; i < end; ++i)
						++local_histogram[in_array[i]];
				
					uint32_t* local_offsets = new uint32_t[buckets_count];
				
					uint32_t sum = 0;
					for (size_t i = 0; i < buckets_count; ++i)
					{
						local_offsets[i] = sum;
						sum += local_histogram[i];
					}
				
					std::sort(in_array + begin, in_array + end);
				
					for (int i = begin; i < end; ++i)
					{
						int digit = in_array[i];
						int local_offset = local_offsets[digit];
				
						// if (wg == 1)
						// 	std::cout << i << ":\t" << dataFromBuffer[i] << std::endl;
				
						if (local_offset != dataFromBuffer[i])
						{
							__debugbreak();
						}
					}
				
					delete[] local_offsets;
					delete[] local_histogram;
				}
#endif
		// if (dataFromBuffer)
		// {
		// 	std::sort(in_array, in_array + in_count);
		// 
		// 	for (int i = 0; i < in_count; ++i)
		// 	{
		// 		if (in_array[i] != dataFromBuffer[i])
		// 			__debugbreak();
		// 	}
		// 
		// 	std::cout << "--------------------" << std::endl;
		// 	std::cout << "PASS" << std::endl;
		// }
#endif

		driver->endScene();

		delete[] in_array;

#if STRESS_TEST
	}
#endif

	return 0;
}