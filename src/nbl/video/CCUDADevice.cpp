// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/video/CUDAInterop.h"

namespace nbl::video
{

CCUDADevice::E_VIRTUAL_ARCHITECTURE CCUDADevice::getVirtualArchitecture() const
{
	return m_virtualArchitecture;
}

core::SRange<const char* const> CCUDADevice::geDefaultCompileOptions() const
{
	return {m_defaultCompileOptions.data(),m_defaultCompileOptions.data()+m_defaultCompileOptions.size()};
}

const CCUDAHandler* CCUDADevice::getHandler() const
{
	return m_handler.get();
}

}

#ifdef _NBL_COMPILE_WITH_CUDA_
#include "CUDAInteropNativeState.hpp"

#ifdef _WIN32
#include <winternl.h>
#endif

namespace nbl::video
{

namespace
{

constexpr const char* VirtualArchCompileOption[] = {
	"-arch=compute_30",
	"-arch=compute_32",
	"-arch=compute_35",
	"-arch=compute_37",
	"-arch=compute_50",
	"-arch=compute_52",
	"-arch=compute_53",
	"-arch=compute_60",
	"-arch=compute_61",
	"-arch=compute_62",
	"-arch=compute_70",
	"-arch=compute_72",
	"-arch=compute_75",
	"-arch=compute_80",
	"-arch=compute_86",
	"-arch=compute_87",
	"-arch=compute_88",
	"-arch=compute_89",
	"-arch=compute_90",
	"-arch=compute_90a",
	"-arch=compute_100",
	"-arch=compute_100a",
	"-arch=compute_100f",
	"-arch=compute_103",
	"-arch=compute_103a",
	"-arch=compute_103f",
	"-arch=compute_110",
	"-arch=compute_110a",
	"-arch=compute_110f",
	"-arch=compute_120",
	"-arch=compute_120a",
	"-arch=compute_120f",
	"-arch=compute_121",
	"-arch=compute_121a",
	"-arch=compute_121f",
};

static_assert(sizeof(VirtualArchCompileOption)/sizeof(*VirtualArchCompileOption)==CCUDADevice::EVA_COUNT);

constexpr CUmemAllocationHandleType AllocationHandleType = 
#ifdef _WIN32
	CU_MEM_HANDLE_TYPE_WIN32;
#else
	CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#endif

}

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

	m_defaultCompileOptions.push_back("--std=c++20");
	m_defaultCompileOptions.push_back(VirtualArchCompileOption[m_virtualArchitecture]);
	m_defaultCompileOptions.push_back("-dc");
	m_defaultCompileOptions.push_back("-use_fast_math");

	const auto& cu = m_handler->getCUDAFunctionTable();
	
	if (!handleCudaCall(cu.pcuCtxCreate_v4(&m_native->context, nullptr, 0, m_native->handle), "Fail to create context!"))
		return;
	if (!handleCudaCall(cu.pcuCtxSetCurrent(m_native->context), "Fail to set current context!"))
		return;

	for (uint32_t locationType = 0; locationType < m_allocationGranularity.size(); ++locationType)
	{
#ifdef _WIN32
		OBJECT_ATTRIBUTES metadata = {
			.Length = sizeof(OBJECT_ATTRIBUTES)
		};
#endif

		const auto prop = CUmemAllocationProp{
			.type = CU_MEM_ALLOCATION_TYPE_PINNED,
			.requestedHandleTypes = AllocationHandleType,
			.location = { .type = static_cast<CUmemLocationType>(locationType), .id = m_native->handle },
#ifdef _WIN32
			.win32HandleMetaData = &metadata,
#endif
		};
		if (!handleCudaCall(cu.pcuMemGetAllocationGranularity(&m_allocationGranularity[locationType], &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "Fail to get memory allocation granularity!"))
			return;
	}
	m_valid = true;
}

cuda_interop::SCUdevice CCUDADevice::getInternalObject() const
{
	return m_native->handle;
}

cuda_interop::SCUcontext CCUDADevice::getContext() const
{
	return m_native->context;
}

static bool isDeviceLocal(CUmemLocationType location)
{
	return location==CU_MEM_LOCATION_TYPE_DEVICE;
}

static CUresult reserveAddressAndMapMemory(const CCUDAHandler& handler, CUdevice nativeDevice, CUdeviceptr* outPtr, size_t size, size_t alignment, CUmemLocationType location, CUmemGenericAllocationHandle memory)
{
	const auto& cu = handler.getCUDAFunctionTable();
	
	CUdeviceptr ptr = 0;
	if (const auto err = cu.pcuMemAddressReserve(&ptr, size, alignment, 0, 0); CUDA_SUCCESS != err)
		return err;

	if (const auto err = cu.pcuMemMap(ptr, size, 0, memory, 0); CUDA_SUCCESS != err)
	{
		handler.defaultHandleResult(cu.pcuMemAddressFree(ptr, size));
		return err;
	}
	
	CUmemAccessDesc accessDesc = {
		.location = { .type = location, .id = nativeDevice },
		.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
	};

	if (auto err = cu.pcuMemSetAccess(ptr, size, &accessDesc, 1); CUDA_SUCCESS != err)
	{
		handler.defaultHandleResult(cu.pcuMemUnmap(ptr, size));
		handler.defaultHandleResult(cu.pcuMemAddressFree(ptr, size));
		return err;
	}

	*outPtr = ptr;

	return CUDA_SUCCESS;
}

core::smart_refctd_ptr<CCUDAExportableMemory> CCUDADevice::createExportableMemory(SExportableMemoryCreationParams&& inParams)
{
	const auto handler = getHandler();
	const auto location = static_cast<CUmemLocationType>(inParams.locationType);

	CCUDAExportableMemory::SCachedCreationParams params = {
		.granularSize = roundToGranularity(inParams.locationType, inParams.size),
		.deviceLocal = isDeviceLocal(location)
	};
	if (params.granularSize==0u)
		return nullptr;

	auto& cu = handler->getCUDAFunctionTable();

#ifdef _WIN32
	OBJECT_ATTRIBUTES metadata = {
		.Length = sizeof(OBJECT_ATTRIBUTES)
	};
#endif

	const auto prop = CUmemAllocationProp{
		.type = CU_MEM_ALLOCATION_TYPE_PINNED,
		.requestedHandleTypes = AllocationHandleType,
		.location = { .type = location, .id = m_native->handle },
#ifdef _WIN32
		.win32HandleMetaData = &metadata,
#endif
	};

	auto nativeState = std::make_unique<CCUDAExportableMemory::SNativeState>();

	CUmemGenericAllocationHandle mem;
	if(!handleCudaCall(cu.pcuMemCreate(&mem, params.granularSize, &prop, 0), "Fail to create memory!"))
		return nullptr;
	
	if (!handleCudaCall(cu.pcuMemExportToShareableHandle(&params.externalHandle, mem, prop.requestedHandleTypes, 0), "Fail to export memory!"))
	{
		handleCudaCall(cu.pcuMemRelease(mem), "Fail to release memory!");
		return nullptr;
	}

	if (!handleCudaCall(reserveAddressAndMapMemory(*handler,m_native->handle,&nativeState->ptr, params.granularSize, inParams.alignment, location, mem), "Fail to reserve address and map memory!"))
	{
		handleCudaCall(cu.pcuMemRelease(mem), "Fail to release memory!");

		if (!system::CloseExternalHandle(params.externalHandle))
			logFail("Fail to close exported CUDA memory handle!");

		return nullptr;
	}

	if (!handleCudaCall(cu.pcuMemRelease(mem), "Fail to release memory!"))
	{
		handleCudaCall(cu.pcuMemUnmap(nativeState->ptr, params.granularSize), "Fail to unmap memory!");
		handleCudaCall(cu.pcuMemAddressFree(nativeState->ptr, params.granularSize), "Fail to free memory address!");
		if (!system::CloseExternalHandle(params.externalHandle))
			logFail("Fail to close exported CUDA memory handle!");
		return nullptr;
	}
	
	return CCUDAExportableMemory::create(core::smart_refctd_ptr<CCUDADevice>(this),std::move(params),std::move(nativeState));
}

core::smart_refctd_ptr<CCUDAImportedMemory> CCUDADevice::importExternalMemory(core::smart_refctd_ptr<IDeviceMemoryAllocation>&& memoryAllocation)
{
	if (!memoryAllocation)
	{
		logFail("The memoryAllocation must not be null!");
		return nullptr;
	}

	const auto& cu = m_handler->getCUDAFunctionTable();
	const auto handleTypes = memoryAllocation->getCreationParams().externalHandleTypes;

	if (!handleTypes.hasFlags(CCUDADevice::ExternalMemoryHandleType))
	{
		logFail("Required memory handle type 0x%x not present in allocation handleTypes 0x%x",
			CCUDADevice::ExternalMemoryHandleType, handleTypes.value);		
	  return nullptr;
	}

	const auto externalHandle = memoryAllocation->getExportHandle(CCUDADevice::ExternalMemoryHandleType);

	CUDA_EXTERNAL_MEMORY_HANDLE_DESC extMemDesc = {};
#ifdef _WIN32
	extMemDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
	extMemDesc.handle.win32.handle = externalHandle;
#else
	extMemDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
	extMemDesc.handle.fd = externalHandle;
#endif
	extMemDesc.size = memoryAllocation->getAllocationSize();

	CUexternalMemory cuExtMem;
	if (!handleCudaCall(cu.pcuImportExternalMemory(&cuExtMem, &extMemDesc), "Fail to import external memory!"))
		return nullptr;

	return core::smart_refctd_ptr<CCUDAImportedMemory>(
		new CCUDAImportedMemory(core::smart_refctd_ptr<CCUDADevice>(this),std::move(memoryAllocation),std::make_unique<CCUDAImportedMemory::SNativeState>(cuExtMem)),
		core::dont_grab
	);
}

core::smart_refctd_ptr<CCUDAImportedSemaphore> CCUDADevice::importExternalSemaphore(core::smart_refctd_ptr<ISemaphore>&& semaphore)
{
	if (!semaphore)
	{
		logFail("The semaphore must not be null");
		return nullptr;
	}

	auto& cu = m_handler->getCUDAFunctionTable();
	auto handleType = semaphore->getCreationParams().externalHandleTypes.value;

	if (handleType != CCUDADevice::ExternalSemaphoreHandleType)
	{
		logFail("Required semaphore handle type 0x%x not present in semaphore handleTypes 0x%x", handleType);
		return nullptr;
	}

	CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC desc = {
#ifdef _WIN32
		.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32,
		.handle = {.win32 = {.handle = semaphore->getExportHandle(ExternalSemaphoreHandleType) }},
#else
		.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD,
		.handle = {.fd = semaphore->getExportHandle(ExternalSemaphoreHandleType)}
#endif
	};


	CUexternalSemaphore cusema;
	if (!handleCudaCall(cu.pcuImportExternalSemaphore(&cusema, &desc), "Fail to import external semaphore!"))
		return nullptr;
	
	return core::smart_refctd_ptr<CCUDAImportedSemaphore>(
		new CCUDAImportedSemaphore(core::smart_refctd_ptr<CCUDADevice>(this),std::move(semaphore),std::make_unique<CCUDAImportedSemaphore::SNativeState>(cusema)),
		core::dont_grab
	);
}

CCUDADevice::~CCUDADevice()
{
	if (m_native->context)
		handleCudaCall(m_handler->getCUDAFunctionTable().pcuCtxDestroy_v2(m_native->context), "Fail to destroy context!");
}

bool CCUDADevice::isValid() const
{
	return m_valid;
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
	, m_virtualArchitecture(virtualArchitecture)
	, m_valid(false)
	, m_handler(std::move(handler))
	, m_native(std::move(nativeState))
{
	assert(false);
}

CCUDADevice::~CCUDADevice() = default;

bool CCUDADevice::isValid() const
{
	return false;
}

cuda_interop::SCUdevice CCUDADevice::getInternalObject() const
{
	return {};
}

cuda_interop::SCUcontext CCUDADevice::getContext() const
{
	return {};
}

core::smart_refctd_ptr<CCUDAExportableMemory> CCUDADevice::createExportableMemory(SExportableMemoryCreationParams&&)
{
	return nullptr;
}

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
