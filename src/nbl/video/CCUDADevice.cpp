// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/video/CUDAInterop.h"

#ifdef _NBL_COMPILE_WITH_CUDA_
#include "CUDAInteropNativeState.hpp"

#ifdef _WIN32
#include <winternl.h>
#endif

namespace nbl::video
{

CCUDADevice::CCUDADevice(
	core::smart_refctd_ptr<CVulkanConnection>&& vulkanConnection, 
	IPhysicalDevice* const vulkanDevice, 
	const E_VIRTUAL_ARCHITECTURE virtualArchitecture,
	std::unique_ptr<SNativeState>&& nativeState,
	core::smart_refctd_ptr<CCUDAHandler>&& handler) : 
	m_logger(vulkanDevice->getDebugCallback()->getLogger()),
  m_defaultCompileOptions(), 
  m_vulkanConnection(std::move(vulkanConnection)), 
  m_physicalDevice(vulkanDevice), 
  m_virtualArchitecture(virtualArchitecture),
	m_handler(std::move(handler)),
	m_native(std::move(nativeState))
{
	assert(m_native);

	m_defaultCompileOptions.push_back("--std=c++14");
	m_defaultCompileOptions.push_back(virtualArchCompileOption[m_virtualArchitecture]);
	m_defaultCompileOptions.push_back("-dc");
	m_defaultCompileOptions.push_back("-use_fast_math");

  const auto& cu = cuda_native::CCUDAHandlerAccessor::getCUDAFunctionTable(*m_handler);
	
	if (!cuda_native::CCUDAHandlerAccessor::defaultHandleResult(*m_handler, cu.pcuCtxCreate_v4(&m_native->context, nullptr, 0, m_native->handle)))
		assert(false);
	if (!cuda_native::CCUDAHandlerAccessor::defaultHandleResult(*m_handler, cu.pcuCtxSetCurrent(m_native->context)))
		assert(false);

	for (uint32_t locationType = 0; locationType < m_native->allocationGranularity.size(); ++locationType)
	{
	
    #ifdef _WIN32
      OBJECT_ATTRIBUTES metadata = {
        .Length = sizeof(OBJECT_ATTRIBUTES)
      };
    #endif

	  const auto prop = CUmemAllocationProp{
      .type = CU_MEM_ALLOCATION_TYPE_PINNED,
      .requestedHandleTypes = cuda_native::SAccess::allocationHandleType(),
      .location = { .type = static_cast<CUmemLocationType>(locationType), .id = m_native->handle },
  #ifdef _WIN32
      .win32HandleMetaData = &metadata,
  #endif
    };
		if (!cuda_native::CCUDAHandlerAccessor::defaultHandleResult(*m_handler, cu.pcuMemGetAllocationGranularity(&m_native->allocationGranularity[locationType], &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM)))
			assert(false);
	}
}

namespace cuda_native
{

CUdevice CCUDADeviceAccessor::getInternalObject(const CCUDADevice& device)
{
	return SAccess::native(device).handle;
}

CUcontext CCUDADeviceAccessor::getContext(const CCUDADevice& device)
{
	return SAccess::native(device).context;
}

size_t CCUDADeviceAccessor::roundToGranularity(const CCUDADevice& device, CUmemLocationType location, size_t size)
{
	const auto& granularity = SAccess::native(device).allocationGranularity[location];
	return ((size - 1) / granularity + 1) * granularity;
}

}

static bool isDeviceLocal(CUmemLocationType location)
{
	return location==CU_MEM_LOCATION_TYPE_DEVICE;
}

static CUresult reserveAddressAndMapMemory(const CCUDADevice& device, CUdeviceptr* outPtr, size_t size, size_t alignment, CUmemLocationType location, CUmemGenericAllocationHandle memory)
{
	const auto handler = device.getHandler();
	const auto& native = cuda_native::SAccess::native(device);
	const auto& cu = cuda_native::CCUDAHandlerAccessor::getCUDAFunctionTable(*handler);
	
	CUdeviceptr ptr = 0;
	if (const auto err = cu.pcuMemAddressReserve(&ptr, size, alignment, 0, 0); CUDA_SUCCESS != err)
		return err;

	if (const auto err = cu.pcuMemMap(ptr, size, 0, memory, 0); CUDA_SUCCESS != err)
	{
		if (!cuda_native::CCUDAHandlerAccessor::defaultHandleResult(*handler, cu.pcuMemAddressFree(ptr, size)))
			assert(false);
		return err;
	}
	
	CUmemAccessDesc accessDesc = {
		.location = { .type = location, .id = native.handle },
		.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
	};

	if (auto err = cu.pcuMemSetAccess(ptr, size, &accessDesc, 1); CUDA_SUCCESS != err)
	{
		if (!cuda_native::CCUDAHandlerAccessor::defaultHandleResult(*handler, cu.pcuMemUnmap(ptr, size)))
			assert(false);
		if (!cuda_native::CCUDAHandlerAccessor::defaultHandleResult(*handler, cu.pcuMemAddressFree(ptr, size)))
			assert(false);
		return err;
	}

	*outPtr = ptr;

	return CUDA_SUCCESS;
}

namespace cuda_native
{

core::smart_refctd_ptr<CCUDAExportableMemory> CCUDADeviceAccessor::createExportableMemory(CCUDADevice& device, SExportableMemoryCreationParams&& inParams)
{
	const auto handler = device.getHandler();
	auto& native = SAccess::native(device);
	auto logger = SAccess::logger(device);

	CCUDAExportableMemory::SCachedCreationParams params = {
		.size = inParams.size,
		.alignment = inParams.alignment,
		.granularSize = CCUDADeviceAccessor::roundToGranularity(device, inParams.location, inParams.size),
		.deviceLocal = isDeviceLocal(inParams.location)
	};

	auto& cu = CCUDAHandlerAccessor::getCUDAFunctionTable(*handler);
	
#ifdef _WIN32
	OBJECT_ATTRIBUTES metadata = {
	  .Length = sizeof(OBJECT_ATTRIBUTES)
	};
#endif

	 const auto prop = CUmemAllocationProp{
		.type = CU_MEM_ALLOCATION_TYPE_PINNED,
		.requestedHandleTypes = SAccess::allocationHandleType(),
		.location = { .type = inParams.location, .id = native.handle },
#ifdef _WIN32
		.win32HandleMetaData = &metadata,
#endif
	};

	auto nativeState = SAccess::makeExportableMemoryNativeState();

	CUmemGenericAllocationHandle mem;
	if(auto err = cu.pcuMemCreate(&mem, params.granularSize, &prop, 0); CUDA_SUCCESS != err)
	{
		logger.log("Fail to create memory handle!", system::ILogger::ELL_ERROR);
		return nullptr;
	}
	
	if (auto err = cu.pcuMemExportToShareableHandle(&params.externalHandle, mem, prop.requestedHandleTypes, 0); CUDA_SUCCESS != err)
	{
		logger.log("Fail to create externalHandle!", system::ILogger::ELL_ERROR);
		if (!CCUDAHandlerAccessor::defaultHandleResult(*handler, cu.pcuMemRelease(mem)))
			assert(false);
		return nullptr;
	}

	if (const auto err = reserveAddressAndMapMemory(device,&SAccess::deviceptr(*nativeState), params.granularSize, params.alignment, inParams.location, mem); CUDA_SUCCESS != err)
	{
		logger.log("Fail to reserve address and map memory!", system::ILogger::ELL_ERROR);

		if (!CCUDAHandlerAccessor::defaultHandleResult(*handler, cu.pcuMemRelease(mem)))
			assert(false);

		bool closeSucceed = CloseExternalHandle(params.externalHandle);
		assert(closeSucceed);

		return nullptr;
	}

	if (const auto err = cu.pcuMemRelease(mem); CUDA_SUCCESS != err)
	{
		bool closeSucceed = CloseExternalHandle(params.externalHandle);
		assert(closeSucceed);
		return nullptr;
	}
	
	return SAccess::makeExportableMemory(core::smart_refctd_ptr<CCUDADevice>(&device),std::move(params),std::move(nativeState));
}

}

core::smart_refctd_ptr<CCUDAImportedMemory> CCUDADevice::importExternalMemory(core::smart_refctd_ptr<IDeviceMemoryAllocation>&& mem)
{
	const auto& cu = cuda_native::CCUDAHandlerAccessor::getCUDAFunctionTable(*m_handler);
	const auto handleType = mem->getCreationParams().externalHandleType;

	if (!handleType) return nullptr;

	const auto externalHandle = mem->getExternalHandle();

	CUDA_EXTERNAL_MEMORY_HANDLE_DESC extMemDesc = {};
#ifdef _WIN32
	extMemDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
	extMemDesc.handle.win32.handle = externalHandle;
#else
	extMemDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
	extMemDesc.handle.fd = externalHandle;
#endif
	extMemDesc.size = mem->getAllocationSize();

	CUexternalMemory cuExtMem;
	if (const auto err = cu.pcuImportExternalMemory(&cuExtMem, &extMemDesc); CUDA_SUCCESS != err)
	{
		m_logger.log("Fail to import external memory into CUDA!", system::ILogger::ELL_ERROR);
		return nullptr;
	}
	return core::smart_refctd_ptr<CCUDAImportedMemory>(
		new CCUDAImportedMemory(core::smart_refctd_ptr<CCUDADevice>(this),std::move(mem),std::make_unique<CCUDAImportedMemory::SNativeState>(cuExtMem)),
		core::dont_grab
	);
}

core::smart_refctd_ptr<CCUDAImportedSemaphore> CCUDADevice::importExternalSemaphore(core::smart_refctd_ptr<ISemaphore>&& sema)
{
	auto& cu = cuda_native::CCUDAHandlerAccessor::getCUDAFunctionTable(*m_handler);
	auto handleType = sema->getCreationParams().externalHandleTypes.value;

	if (!handleType)
		return nullptr;

	CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC desc = {
#ifdef _WIN32
		.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32,
		// TODO(kevinyu): Fix this later. Make it compile first.
		.handle = {.win32 = {.handle = sema->getExternalHandle() }},
#else
    .type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD,
		.handle = {.fd = sema->getExternalHandle()}
#endif
	};


	CUexternalSemaphore cusema;
	if (const auto err = cu.pcuImportExternalSemaphore(&cusema, &desc); CUDA_SUCCESS != err)
	{
		m_logger.log("Fail to import semaphore into CUDA!");
		return nullptr;
	}
	
	return core::smart_refctd_ptr<CCUDAImportedSemaphore>(
		new CCUDAImportedSemaphore(core::smart_refctd_ptr<CCUDADevice>(this),std::move(sema),std::make_unique<CCUDAImportedSemaphore::SNativeState>(cusema)),
		core::dont_grab
	);
}

CCUDADevice::~CCUDADevice()
{
	if (!cuda_native::CCUDAHandlerAccessor::defaultHandleResult(*m_handler, cuda_native::CCUDAHandlerAccessor::getCUDAFunctionTable(*m_handler).pcuCtxDestroy_v2(m_native->context)))
		assert(false);
}

}

#else

namespace nbl::video
{

// CUDA OFF stub keeps the clean public API linkable and reports feature absence with nullptr instead of unresolved symbols.
struct CCUDADevice::SNativeState {};

CCUDADevice::CCUDADevice(
	core::smart_refctd_ptr<CVulkanConnection>&& vulkanConnection,
	IPhysicalDevice* const vulkanDevice,
	const E_VIRTUAL_ARCHITECTURE virtualArchitecture,
	std::unique_ptr<SNativeState>&& nativeState,
	core::smart_refctd_ptr<CCUDAHandler>&& handler)
	: m_logger(nullptr)
	, m_vulkanConnection(std::move(vulkanConnection))
	, m_physicalDevice(vulkanDevice)
	, m_virtualArchitecture(virtualArchitecture)
	, m_handler(std::move(handler))
	, m_native(std::move(nativeState))
{
	assert(m_native);
}

CCUDADevice::~CCUDADevice() = default;

core::smart_refctd_ptr<CCUDAImportedMemory> CCUDADevice::importExternalMemory(core::smart_refctd_ptr<IDeviceMemoryAllocation>&&)
{
	return nullptr;
}

core::smart_refctd_ptr<CCUDAImportedSemaphore> CCUDADevice::importExternalSemaphore(core::smart_refctd_ptr<ISemaphore>&&)
{
	return nullptr;
}

}

#endif // _NBL_COMPILE_WITH_CUDA_
