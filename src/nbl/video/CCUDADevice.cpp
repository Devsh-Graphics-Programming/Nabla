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
	core::smart_refctd_ptr<CVulkanConnection>&& _vulkanConnection, 
	IPhysicalDevice* const _vulkanDevice, 
	const E_VIRTUAL_ARCHITECTURE _virtualArchitecture,
	CUdevice _device,
	core::smart_refctd_ptr<CCUDAHandler>&& _handler) : 
  m_defaultCompileOptions(), 
  m_vulkanConnection(std::move(_vulkanConnection)), 
  m_vulkanDevice(_vulkanDevice), 
  m_virtualArchitecture(_virtualArchitecture),
	m_handle(_device),
	m_handler(std::move(_handler)),
	m_allocationGranularity{}
{
	m_defaultCompileOptions.push_back("--std=c++14");
	m_defaultCompileOptions.push_back(virtualArchCompileOption[m_virtualArchitecture]);
	m_defaultCompileOptions.push_back("-dc");
	m_defaultCompileOptions.push_back("-use_fast_math");

  auto& cu = m_handler->getCUDAFunctionTable();
	
	CUresult re = cu.pcuCtxCreate_v4(&m_context, nullptr, 0, m_handle);
	assert(CUDA_SUCCESS == re);
	re = cu.pcuCtxSetCurrent(m_context);
	assert(CUDA_SUCCESS == re);

	for (uint32_t i = 0; i < ARRAYSIZE(m_allocationGranularity); ++i)
	{
		const auto prop = getMemAllocationProp(static_cast<CUmemLocationType>(i));
		auto re = cu.pcuMemGetAllocationGranularity(&m_allocationGranularity[i], &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

		assert(CUDA_SUCCESS == re);
	}
}

size_t CCUDADevice::roundToGranularity(CUmemLocationType location, size_t size) const
{
	return ((size - 1) / m_allocationGranularity[location] + 1) * m_allocationGranularity[location];
}

CUresult CCUDADevice::reserveAddressAndMapMemory(CUdeviceptr* outPtr, size_t size, size_t alignment, CUmemLocationType location, CUmemGenericAllocationHandle memory) const
{
	auto& cu = m_handler->getCUDAFunctionTable();
	
	CUdeviceptr ptr = 0;
	if (auto err = cu.pcuMemAddressReserve(&ptr, size, alignment, 0, 0); CUDA_SUCCESS != err)
		return err;

	if (auto err = cu.pcuMemMap(ptr, size, 0, memory, 0); CUDA_SUCCESS != err)
	{
		cu.pcuMemAddressFree(ptr, size);
		return err;
	}
	
	CUmemAccessDesc accessDesc = {
		.location = { .type = location, .id = m_handle },
		.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
	};

	if (auto err = cu.pcuMemSetAccess(ptr, size, &accessDesc, 1); CUDA_SUCCESS != err)
	{
		cu.pcuMemUnmap(ptr, size);
		cu.pcuMemAddressFree(ptr, size);
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
		return nullptr;
	
	if (auto err = cu.pcuMemExportToShareableHandle(&params.externalHandle, mem, prop.requestedHandleTypes, 0); CUDA_SUCCESS != err)
	{
		cu.pcuMemRelease(mem);
		return nullptr;
	}

	if (auto err = reserveAddressAndMapMemory(&params.ptr, params.granularSize, params.alignment, params.location, mem); CUDA_SUCCESS != err)
	{
		CloseExternalHandle(params.externalHandle);
		cu.pcuMemRelease(mem);
		return nullptr;
	}

	if (auto err = cu.pcuMemRelease(mem); CUDA_SUCCESS != err)
	{
		CloseExternalHandle(params.externalHandle);
		return nullptr;
	}
	
	return core::make_smart_refctd_ptr<CCUDAExportableMemory>(core::smart_refctd_ptr<CCUDADevice>(this), std::move(params), mem);
}

core::smart_refctd_ptr<CCUDAImportedMemory> CCUDADevice::importExternalMemory(core::smart_refctd_ptr<IDeviceMemoryAllocation>&& mem)
{

	auto& cu = m_handler->getCUDAFunctionTable();
	auto handleType = mem->getCreationParams().externalHandleType;

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
	if (auto err = cu.pcuImportExternalMemory(&cuExtMem, &extMemDesc); CUDA_SUCCESS != err)
		return nullptr;
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
	if (auto err = cu.pcuImportExternalSemaphore(&cusema, &desc); CUDA_SUCCESS != err)
		return nullptr;
	
	return core::make_smart_refctd_ptr<CCUDAImportedSemaphore>(core::smart_refctd_ptr<CCUDADevice>(this), std::move(sema), cusema);
}

CCUDADevice::~CCUDADevice()
{
	m_handler->getCUDAFunctionTable().pcuCtxDestroy_v2(m_context);
}

}

#endif // _NBL_COMPILE_WITH_CUDA_
