// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <numeric>

#include "../../source/Nabla/CMemoryFile.h"

#include "nbl/ext/OptiX/Manager.h"

#include "optix_function_table_definition.h"

using namespace nbl;
using namespace asset;
using namespace video;

using namespace nbl::ext::OptiX;


core::smart_refctd_ptr<Manager> Manager::create(video::IVideoDriver* _driver, io::IFileSystem* _filesystem)
{
	if (!_driver)
		return nullptr;

	cuda::CCUDAHandler::init();

	int32_t version = 0;
	if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::cuda.pcuDriverGetVersion(&version)) || version<7000)
		return nullptr;

	// find device
	uint32_t foundDeviceCount = 0u;
	CUdevice devices[MaxSLI] = {};
	cuda::CCUDAHandler::getDefaultGLDevices(&foundDeviceCount, devices, MaxSLI);

	// create context
	CUcontext contexts[MaxSLI] = {};
	bool ownContext[MaxSLI] = {};
	uint32_t suitableDevices = 0u;
	for (uint32_t i=0u; i<foundDeviceCount; i++)
	{
		if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::cuda.pcuCtxCreate_v2(contexts+suitableDevices, CU_CTX_SCHED_YIELD|CU_CTX_MAP_HOST|CU_CTX_LMEM_RESIZE_TO_MAX, devices[suitableDevices])))
			continue;

		uint32_t version = 0u;
		cuda::CCUDAHandler::cuda.pcuCtxGetApiVersion(contexts[suitableDevices],&version);
		if (version<3020)
		{
			cuda::CCUDAHandler::cuda.pcuCtxDestroy_v2(contexts[suitableDevices]);
			continue;
		}
		cuda::CCUDAHandler::cuda.pcuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1);
		ownContext[suitableDevices++] = true;
	}

	if (!suitableDevices)
		return nullptr;

	auto manager = new Manager(_driver,_filesystem,suitableDevices,contexts,ownContext);
	return core::smart_refctd_ptr<Manager>(manager,core::dont_grab);
}

Manager::Manager(video::IVideoDriver* _driver, io::IFileSystem* _filesystem, uint32_t _contextCount, CUcontext* _contexts, bool* _ownContexts) : driver(_driver)
{
	assert(_contextCount<=MaxSLI);

	// Initialize the OptiX API, loading all API entry points 
	optixInit();

	for (uint32_t i=0u; i<MaxSLI; i++)
	{
		context[i] = _contexts[i];
		ownContext[i] = _ownContexts ? _ownContexts[i]:false;
		if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::cuda.pcuStreamCreate(stream+i,CU_STREAM_NON_BLOCKING)))
		{
			context[i] = nullptr;
			ownContext[i] = false;
			stream[i] = false;
			continue;
		}
	}

	// TODO: This cannot stay like that, we need a resource compiler to "build-in" the optix CUDA/device headers into nbl::ext::OptiX so that we can retrieve them.
	auto sdkDir = io::path(OPTIX_INCLUDE_DIR)+"/";
	auto addHeader = [&](const char* subpath) -> void
	{
		auto file = _filesystem->createAndOpenFile(sdkDir+subpath);
		if (!file)
			return;

		auto size = file->getSize();

		io::CMemoryReadFile::allocator_type alloc;
		auto data = alloc.allocate(size+1ull);
		file->read(data,size);
		data[size] = 0;
		file->drop();

		auto memfile = core::make_smart_refctd_ptr<io::CMemoryReadFile>(data,size,subpath,core::adopt_memory);
		optixHeaderContents.push_back(reinterpret_cast<const char*>(memfile->getData()));
		optixHeaderNames.push_back(memfile->getFileName().c_str());
		optixHeaders.push_back(core::move_and_static_cast<io::IReadFile>(memfile));
	};
	addHeader("optix.h");
	addHeader("optix_device.h");
	addHeader("optix_7_device.h");
	addHeader("optix_7_types.h");
	addHeader("internal/optix_7_device_impl.h");
	addHeader("internal/optix_7_device_impl_exception.h");
	addHeader("internal/optix_7_device_impl_transformations.h");

	auto range = cuda::CCUDAHandler::getCUDASTDHeaders();
	for (auto header : range)
		optixHeaders.push_back(core::smart_refctd_ptr<const io::IReadFile>(header));
	optixHeaderContents.insert(optixHeaderContents.end(),cuda::CCUDAHandler::getCUDASTDHeaderContents().begin(),cuda::CCUDAHandler::getCUDASTDHeaderContents().end());
    optixHeaderNames.insert(optixHeaderNames.end(),cuda::CCUDAHandler::getCUDASTDHeaderNames().begin(),cuda::CCUDAHandler::getCUDASTDHeaderNames().end());
}

Manager::~Manager()
{
	optixHeaders.clear();
	optixHeaderContents.clear();
	optixHeaderNames.clear();

	for (uint32_t i=0u; i<MaxSLI; i++)
	{
		if (!context[i])
			continue;
		if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::cuda.pcuCtxPushCurrent_v2(context[i])))
			continue;
		cuda::CCUDAHandler::cuda.pcuCtxSynchronize();

		cuda::CCUDAHandler::cuda.pcuStreamDestroy_v2(stream[i]);
		if (ownContext[i])
			cuda::CCUDAHandler::cuda.pcuCtxDestroy_v2(context[i]);
	}
}


void Manager::defaultCallback(unsigned int level, const char* tag, const char* message, void* cbdata)
{
	uint32_t contextID = reinterpret_cast<const uint32_t&>(cbdata);
	printf("nbl::ext::OptiX Context:%d [%s]: %s\n", contextID, tag, message);
}

