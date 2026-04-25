// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/video/CCUDADevice.h"

#ifdef _WIN32
#include <winternl.h>
#endif

#include "nbl/video/CCUDAImportedMemory.h"

#ifdef _NBL_COMPILE_WITH_CUDA_
namespace nbl::video
{

CCUDADevice::CCUDADevice(
	core::smart_refctd_ptr<CVulkanConnection>&& vulkanConnection, 
	IPhysicalDevice* const vulkanDevice, 
	const E_VIRTUAL_ARCHITECTURE virtualArchitecture,
	CUdevice device,
	core::smart_refctd_ptr<CCUDAHandler>&& handler) : 
	m_logger(vulkanDevice->getDebugCallback()->getLogger()),
  m_defaultCompileOptions(), 
  m_vulkanConnection(std::move(vulkanConnection)), 
  m_physicalDevice(vulkanDevice), 
  m_virtualArchitecture(virtualArchitecture),
	m_handle(device),
	m_handler(std::move(handler)),
	m_allocationGranularity{}
{
	m_defaultCompileOptions.push_back("--std=c++14");
	m_defaultCompileOptions.push_back(virtualArchCompileOption[m_virtualArchitecture]);
	m_defaultCompileOptions.push_back("-dc");
	m_defaultCompileOptions.push_back("-use_fast_math");

  const auto& cu = m_handler->getCUDAFunctionTable();
	
	ASSERT_CUDA_SUCCESS(cu.pcuCtxCreate_v4(&m_context, nullptr, 0, m_handle), m_handler);
	ASSERT_CUDA_SUCCESS(cu.pcuCtxSetCurrent(m_context), m_handler);

	for (uint32_t locationType = 0; locationType < m_allocationGranularity.size(); ++locationType)
	{
		const auto prop = getMemAllocationProp(static_cast<CUmemLocationType>(locationType));
		ASSERT_CUDA_SUCCESS(cu.pcuMemGetAllocationGranularity(&m_allocationGranularity[locationType], &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM), m_handler);
	}
}

size_t CCUDADevice::roundToGranularity(CUmemLocationType location, size_t size) const
{
	return ((size - 1) / m_allocationGranularity[location] + 1) * m_allocationGranularity[location];
}

CUresult CCUDADevice::reserveAddressAndMapMemory(CUdeviceptr* outPtr, size_t size, size_t alignment, CUmemLocationType location, CUmemGenericAllocationHandle memory) const
{
	const auto& cu = m_handler->getCUDAFunctionTable();
	
	CUdeviceptr ptr = 0;
	if (const auto err = cu.pcuMemAddressReserve(&ptr, size, alignment, 0, 0); CUDA_SUCCESS != err)
		return err;

	if (const auto err = cu.pcuMemMap(ptr, size, 0, memory, 0); CUDA_SUCCESS != err)
	{
		ASSERT_CUDA_SUCCESS(cu.pcuMemAddressFree(ptr, size), m_handler);
		return err;
	}
	
	CUmemAccessDesc accessDesc = {
		.location = { .type = location, .id = m_handle },
		.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
	};

	if (auto err = cu.pcuMemSetAccess(ptr, size, &accessDesc, 1); CUDA_SUCCESS != err)
	{
		ASSERT_CUDA_SUCCESS(cu.pcuMemUnmap(ptr, size), m_handler);
		ASSERT_CUDA_SUCCESS(cu.pcuMemAddressFree(ptr, size), m_handler);
		return err;
	}

	*outPtr = ptr;

	return CUDA_SUCCESS;
}

CUmemAllocationProp CCUDADevice::getMemAllocationProp(CUmemLocationType locationType) const
{
	
#ifdef _WIN32
	OBJECT_ATTRIBUTES metadata = {};
	metadata.Length = sizeof(OBJECT_ATTRIBUTES);
#endif

	 return {
		.type = CU_MEM_ALLOCATION_TYPE_PINNED,
		.requestedHandleTypes = ALLOCATION_HANDLE_TYPE,
		.location = { .type = locationType, .id = m_handle },
#ifdef _WIN32
		.win32HandleMetaData = &metadata,
#endif
	};
}

core::smart_refctd_ptr<CCUDAExportableMemory> CCUDADevice::createExportableMemory(CCUDAExportableMemory::SCreationParams&& inParams)
{
	CCUDAExportableMemory::SCachedCreationParams params = { inParams };

	auto& cu = m_handler->getCUDAFunctionTable();

	const auto prop = getMemAllocationProp(params.location);
	
	params.granularSize = roundToGranularity(params.location, params.size);

	CUmemGenericAllocationHandle mem;
	if(auto err = cu.pcuMemCreate(&mem, params.granularSize, &prop, 0); CUDA_SUCCESS != err)
	{
		m_logger.log("Fail to create memory handle!", system::ILogger::ELL_ERROR);
		return nullptr;
	}
	
	if (auto err = cu.pcuMemExportToShareableHandle(&params.externalHandle, mem, prop.requestedHandleTypes, 0); CUDA_SUCCESS != err)
	{
		m_logger.log("Fail to create externalHandle!", system::ILogger::ELL_ERROR);
		ASSERT_CUDA_SUCCESS(cu.pcuMemRelease(mem), m_handler);
		return nullptr;
	}

	if (const auto err = reserveAddressAndMapMemory(&params.ptr, params.granularSize, params.alignment, params.location, mem); CUDA_SUCCESS != err)
	{
		m_logger.log("Fail to reserve address and map memory!", system::ILogger::ELL_ERROR);

		ASSERT_CUDA_SUCCESS(cu.pcuMemRelease(mem), m_handler);

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
	
	return core::make_smart_refctd_ptr<CCUDAExportableMemory>(core::smart_refctd_ptr<CCUDADevice>(this), std::move(params));
}

core::smart_refctd_ptr<CCUDAImportedMemory> CCUDADevice::importExternalMemory(core::smart_refctd_ptr<IDeviceMemoryAllocation>&& mem)
{
	const auto& cu = m_handler->getCUDAFunctionTable();
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
	return core::make_smart_refctd_ptr<CCUDAImportedMemory>(core::smart_refctd_ptr<CCUDADevice>(this), std::move(mem), cuExtMem);
}

core::smart_refctd_ptr<CCUDAImportedSemaphore> CCUDADevice::importExternalSemaphore(core::smart_refctd_ptr<ISemaphore>&& sema)
{
	auto& cu = m_handler->getCUDAFunctionTable();
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
	
	return core::make_smart_refctd_ptr<CCUDAImportedSemaphore>(core::smart_refctd_ptr<CCUDADevice>(this), std::move(sema), cusema);
}

CCUDADevice::~CCUDADevice()
{
	ASSERT_CUDA_SUCCESS(m_handler->getCUDAFunctionTable().pcuCtxDestroy_v2(m_context), m_handler);
}

}

#endif // _NBL_COMPILE_WITH_CUDA_
