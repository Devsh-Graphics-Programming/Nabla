#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../../source/Nabla/COpenGLDriver.h"

using namespace nbl;
using namespace core;
using namespace video;
using namespace asset;

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

	const size_t in_count = 147;
	const size_t in_size = in_count * sizeof(int32_t);
	int32_t in[in_count] = { -32518, -32702, -32740, -32435, -31619, -32752, -32233, -32730, -32698,
		-32468, -32582, -32556, -32525, -32615, -32475, -32718, -32024, -32457, -32467, -32663, -32735,
		-32524, -32666, -32397, -32684, -32088, -32683, -32741, -32761, -32715, -32730, -32405, -32588,
		-32610, -31794, -32622, -32711, -32324, -32425, -32692, -32604, -32660, -32761, -32685, -32572,
		-32450, -32636, -32674, -32661, -32684, -32399, -32194, -32536, -32575, -32605, -32630, -32689,
		-32743, -32130, -32554, -32737, -32534, -32696, -31740, -32733, -32326, -32625, -32603, -32554,
		-32756, -32582, -32592, -32750, -32464, -32649, -32396, -32645, -32032, -32278, -32179, -32710,
		-32372, -32418, -32597, -32748, -32761, -32722, -32368, -32658, -32621, -32672, -32661, -32726,
		-32632, -32474, -32713, -31854, -32682, -32704, -32126, -32486, -32279, -32131, -32613, -30809,
		-32686, -32728, -32723, -32705, -32369, -32704, -31879, -32529, -32350, -32544, -32726, -32724,
		-32424, -32725, -32149, -32515, -32705, -32519, -32660, -32687, -32519, -32446, -32342, -32716,
		-32629, -32733, -32464, -32749, -32745, -32532, -31924, -32737, -32570, -32402, -32571, -32350,
		-31861, -32631, -32645, -32726, -32734, -32672 };



	auto in_gpu = driver->createFilledDeviceLocalGPUBufferOnDedMem(in_size, in);

	smart_refctd_ptr<IGPUComputePipeline> pipeline = nullptr;
	smart_refctd_ptr<IGPUDescriptorSet> ds = nullptr;
	{
		const uint32_t count = 1u;
		IGPUDescriptorSetLayout::SBinding binding[count];
		for (uint32_t i = 0; i < count; ++i)
			binding[i] = { i, asset::EDT_STORAGE_BUFFER, 1u, IGPUSpecializedShader::ESS_COMPUTE, nullptr };

		auto ds_layout_gpu = driver->createGPUDescriptorSetLayout(binding, binding + count);
		ds = driver->createGPUDescriptorSet(smart_refctd_ptr(ds_layout_gpu));

		auto pipeline_layout = driver->createGPUPipelineLayout(nullptr, nullptr, smart_refctd_ptr(ds_layout_gpu));

		smart_refctd_ptr<IGPUSpecializedShader> shader_gpu = nullptr;
		{
			auto file = smart_refctd_ptr<io::IReadFile>(filesystem->createAndOpenFile("../Debug.comp"));

			asset::IAssetLoader::SAssetLoadParams lp;
			auto cs_bundle = am->getAsset("../Debug.comp", lp);
			auto cs = smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*cs_bundle.getContents().begin());
			auto cs_rawptr = cs.get();

			shader_gpu = driver->getGPUObjectsFromAssets(&cs_rawptr, &cs_rawptr + 1)->front();
		}

		pipeline = driver->createGPUComputePipeline(nullptr, std::move(pipeline_layout), std::move(shader_gpu));
	}

	driver->beginScene(true);

	{
		const uint32_t count = 1;
		IGPUDescriptorSet::SDescriptorInfo ds_info[count];
		ds_info[0].desc = in_gpu;
		ds_info[0].buffer = { 0u, in_size };

		IGPUDescriptorSet::SWriteDescriptorSet writes[count];
		for (uint32_t i = 0; i < count; ++i)
			writes[i] = { ds.get(), i, 0u, 1u, asset::EDT_STORAGE_BUFFER, ds_info + i };

		driver->updateDescriptorSets(count, writes, 0u, nullptr);
	}

	driver->bindComputePipeline(pipeline.get());
	driver->bindDescriptorSets(video::EPBP_COMPUTE, pipeline->getLayout(), 0u, 1u, &ds.get(), nullptr);
	driver->dispatch(1, 1, 1);

	video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

	int32_t* debug_cpu = new int32_t[in_count];

	int32_t prefix_scan = INT_MAX;
	for (uint32_t i = 0; i < in_count; ++i)
	{
		debug_cpu[i] = prefix_scan;
		prefix_scan = min(prefix_scan, in[i]);
	}

	DebugCompareGPUvsCPU<int32_t>(in_gpu, debug_cpu, in_size, driver);


	driver->endScene();


	return 0;
}
