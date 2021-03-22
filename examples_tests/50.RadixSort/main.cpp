#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../../source/Nabla/COpenGLDriver.h"

using namespace nbl;
using namespace core;
using namespace video;

struct SortElement { uint32_t key, data; };

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

	const uint32_t wg_dim = 1 << 8;
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
			driver->dispatch(wg_count, 1, 1);

			video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
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

			video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
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

			video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		}
	}
}

//
// Nothing important here!! Just global scan test bed. Gonna remove it before merging and fix the indentation.
// Look at the next `main`.
//

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

	unsigned int seed = 0;

#if 0
	while (true)
	{
#endif
		const size_t in_count = 1 << 10;
		const size_t in_size = in_count * sizeof(uint32_t);
		const size_t wg_dim = 1 << 8;

		uint32_t* in = new uint32_t[in_count];
		srand(seed++);
		for (size_t i = 0u; i < in_count; ++i)
			in[i] = rand();

		auto in_gpu = driver->createFilledDeviceLocalGPUBufferOnDedMem(in_size, in);

		smart_refctd_ptr<IGPUDescriptorSet> ds_prefix_sum = nullptr;
		smart_refctd_ptr<IGPUComputePipeline> prefix_sum_pipeline = nullptr;
		{
			const uint32_t count = 1u;
			IGPUDescriptorSetLayout::SBinding binding[count];
			for (uint32_t i = 0; i < count; ++i)
				binding[i] = { i, asset::EDT_STORAGE_BUFFER, 1u, IGPUSpecializedShader::ESS_COMPUTE, nullptr };

			auto ds_layout = driver->createGPUDescriptorSetLayout(binding, binding + count);
			ds_prefix_sum = driver->createGPUDescriptorSet(smart_refctd_ptr(ds_layout));
			auto pipeline_layout = driver->createGPUPipelineLayout(nullptr, nullptr, smart_refctd_ptr(ds_layout));

			smart_refctd_ptr<IGPUSpecializedShader> shader_gpu = nullptr;
			{
				auto file = smart_refctd_ptr<io::IReadFile>(filesystem->createAndOpenFile("../PrefixSum.comp"));

				asset::IAssetLoader::SAssetLoadParams lp;
				auto cs_bundle = am->getAsset("../PrefixSum.comp", lp);
				auto cs = smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*cs_bundle.getContents().begin());
				auto cs_rawptr = cs.get();

				shader_gpu = driver->getGPUObjectsFromAssets(&cs_rawptr, &cs_rawptr + 1)->front();
			}

			prefix_sum_pipeline = driver->createGPUComputePipeline(nullptr, std::move(pipeline_layout), std::move(shader_gpu));
		}

		driver->beginScene(true);


		{
			const uint32_t count = 1;
			IGPUDescriptorSet::SDescriptorInfo ds_info[count];
			ds_info[0].desc = in_gpu;
			ds_info[0].buffer = { 0u, in_size };

			IGPUDescriptorSet::SWriteDescriptorSet writes[count];
			for (uint32_t i = 0; i < count; ++i)
				writes[i] = { ds_prefix_sum.get(), i, 0u, 1u, asset::EDT_STORAGE_BUFFER, ds_info + i };

			driver->updateDescriptorSets(count, writes, 0u, nullptr);
		}

		driver->bindComputePipeline(prefix_sum_pipeline.get());
		driver->bindDescriptorSets(video::EPBP_COMPUTE, prefix_sum_pipeline->getLayout(), 0u, 1u, &ds_prefix_sum.get(), nullptr);

		// Do all the upsweep passes
		// Lets say we have 256 * 256 * 256 == 2^24 or 256^3 elements
		// all in all pass_count = log_wg_dim(n) - 1  <-- `-1` is there is because I'm doing a top of hierarchy pass separately
		// pass_count = log_wg_dim(n) = log_256(256^2) = 2
		// pass 0: will take in 256^2 elements and spit out 256 elements
		// pass 1: will take in those 256 elements and do a top of hierarchy pass

		size_t wg_count = (in_count + wg_dim - 1) / wg_dim;
		uint32_t pass_count = (log(in_count) / log(wg_dim)) - 1;
		for (uint32_t pass = 0; pass < pass_count; ++pass)
		{
			// For each pass the memory access pattern is going to be different
			// For pass 0: each value gets a thread
			//	==> 0, 1, 2, 3, 4, 5, 6, ...
			// For pass 1: only the last element of each wg, from the previous set of wgs gets a thread
			//	==> 255, 511, 767, 1023, ...
			//		it is very possible that some of the above elements won't exist, but I'm not dealing with those cases right now
			//		which would require padding with identity
			// For pass 2: only the last element of each wg, from the previous set of wgs gets a thread
			//	==> 255, 511, 767, 1023, ... 256 elements like that will get bunched together in the same wg
			// 255 + (256-1)*256 = 256th element
			// 255 + 255 * 256
			// 
			// pass0: start at 0 and have a difference of 1, so 0, 1, 2, 3, ..
			// pass1: start at 255 and have a difference of 256, so, 255, 511, 767, 1023, ..
			// pass2: start at 255 + 255 * 256 and have a difference of 256*256, soooooo

			// So the memory access pattern should be like:
			// a = 256^(pass) - 1, d = 256^(pass)

			driver->dispatch(wg_count, 1, 1);

			video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		}

		// To test the result of the passes just check if the last value is the reduction of the elements



		// Here you do the next pass

		wg_count = (wg_count + wg_dim - 1) / wg_dim;
		
		driver->dispatch(wg_count, 1, 1);

		video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);



		// Testing

		uint32_t* data_from_buffer = DebugGPUBufferDownload<uint32_t>(in_gpu, in_size, driver);
		if (data_from_buffer)
		{
			uint32_t reductions[4] = { 0 };
			for (int wg = 0; wg < 4; ++wg)
			{
				size_t begin = wg * wg_dim;
				size_t end = (wg + 1) * wg_dim;

				for (size_t i = begin; i < end; ++i)
					reductions[wg] += in[i];
			}

			uint32_t ps_reductions[4] = { 0 };
			
			uint32_t sum = 0;
			for (int i = 0; i < 4; ++i)
			{
				ps_reductions[i] = sum;
				sum += reductions[i];
			}
			
			for (int i = 0; i < 4; ++i)
			{
				if (ps_reductions[i] != data_from_buffer[wg_dim * (i + 1) - 1])
					__debugbreak();
			
				// std::cout << i << ":\t" << data_from_buffer[wg_dim * i - 1] << std::endl;
			}
				// uint32_t ref[wg_dim] = {};
				// 
				// uint32_t k = 0;
				// uint32_t sum = 0;
				// for (int i = begin; i < end; ++i)
				// {
				// 	sum += in[i];
				// 	ref[k++] = sum;
				// }
				// 
				// k = 0;
				// for (int i = begin; i < end; ++i)
				// {
				// 	if (ref[k++] != data_from_buffer[i])
				// 		__debugbreak();
				// }
			// }

			std::cout << "PASS" << std::endl;
		}


		driver->endScene();

		delete[] in;
#if 0
	}
#endif

	return 0;
}
#endif

#if 1

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

	unsigned int seed = 0;

	// Stress Test
	// Note: This stupid thing will be removed before merging
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
		const int wg_count = in_count / wg_dim;

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
			
			ExclusiveSumScanGPU(histogram_gpu, histogram_count, device);
			
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
					writes[i] = { ds_sort_and_scatter.get(), i, 0u, 1u, asset::EDT_STORAGE_BUFFER, ds_info + i };
			
				driver->updateDescriptorSets(count, writes, 0u, nullptr);
			}
			
			driver->bindComputePipeline(sort_and_scatter_pipeline.get());
			driver->bindDescriptorSets(video::EPBP_COMPUTE, sort_and_scatter_pipeline->getLayout(), 0u, 1u, &ds_sort_and_scatter.get(),
				nullptr);
			
			driver->pushConstants(sort_and_scatter_pipeline->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(uint32_t),
				&shift);
			driver->dispatch(wg_count, 1, 1);
			
			video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		}

		// Testing
		
		std::stable_sort(in_array, in_array + in_count, [](const SortElement& a, const SortElement& b) { return a.key < b.key; });

		SortElement* dataFromBuffer = DebugGPUBufferDownload<SortElement>(in_array_gpu, in_size, driver);

		if (dataFromBuffer)
		{
			for (int i = 0; i < in_count; ++i)
			{
				if ((dataFromBuffer[i].key != in_array[i].key) || (dataFromBuffer[i].data != in_array[i].data))
					__debugbreak();
			}
			
			std::cout << "PASS" << std::endl;
		}
		driver->endScene();

		delete[] in_array;
#if 1
	}
#endif

	return 0;
}
#endif