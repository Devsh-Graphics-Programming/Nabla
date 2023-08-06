// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/video/CCUDAHandler.h"

#ifdef _NBL_COMPILE_WITH_CUDA_
namespace nbl::video
{

CCUDADevice::CCUDADevice(core::smart_refctd_ptr<CVulkanConnection>&& _vulkanConnection, IPhysicalDevice* const _vulkanDevice, const E_VIRTUAL_ARCHITECTURE _virtualArchitecture, CUdevice _handle, core::smart_refctd_ptr<CCUDAHandler>&& _handler)
	: m_defaultCompileOptions(), m_vulkanConnection(std::move(_vulkanConnection)), m_vulkanDevice(_vulkanDevice), m_virtualArchitecture(_virtualArchitecture), m_handle(_handle), m_handler(std::move(_handler)), m_allocationGranularity{}
{
	m_defaultCompileOptions.push_back("--std=c++14");
	m_defaultCompileOptions.push_back(virtualArchCompileOption[m_virtualArchitecture]);
	m_defaultCompileOptions.push_back("-dc");
	m_defaultCompileOptions.push_back("-use_fast_math");
	auto& cu = m_handler->getCUDAFunctionTable();
	
	cu.pcuCtxCreate_v2(&m_context, 0, m_handle);

	for (uint32_t i = 0; i < ARRAYSIZE(m_allocationGranularity); ++i)
	{
		uint32_t metaData[16] = { 48 };
		CUmemAllocationProp prop = {
			.type = CU_MEM_ALLOCATION_TYPE_PINNED,
			.requestedHandleTypes = ALLOCATION_HANDLE_TYPE,
			.location = {.type = static_cast<CUmemLocationType>(i), .id = m_handle },
			.win32HandleMetaData = metaData,
		};
		auto re = cu.pcuMemGetAllocationGranularity(&m_allocationGranularity[i], &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

		assert(CUDA_SUCCESS == re);
	}

}

CCUDADevice::~CCUDADevice()
{
	m_handler->getCUDAFunctionTable().pcuCtxDestroy_v2(m_context);
}

size_t CCUDADevice::roundToGranularity(CUmemLocationType location, size_t size) const
{
	return ((size - 1) / m_allocationGranularity[location] + 1) * m_allocationGranularity[location];
}

CUresult CCUDADevice::reserveAdrressAndMapMemory(CUdeviceptr* outPtr, size_t size, size_t alignment, CUmemLocationType location, CUmemGenericAllocationHandle memory)
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

CUresult CCUDADevice::createSharedMemory(
	core::smart_refctd_ptr<CCUDASharedMemory>* outMem, 
	CCUDASharedMemory::SCreationParams&& inParams)
{
	if (!outMem)
		return CUDA_ERROR_INVALID_VALUE;

	CCUDASharedMemory::SCachedCreationParams params = { inParams };

	auto& cu = m_handler->getCUDAFunctionTable();

	uint32_t metaData[16] = { 48 };

	CUmemAllocationProp prop = {
		.type = CU_MEM_ALLOCATION_TYPE_PINNED,
		.requestedHandleTypes = ALLOCATION_HANDLE_TYPE,
		.location = { .type = params.location, .id = m_handle },
		.win32HandleMetaData = metaData,
	};
	
	params.granularSize = roundToGranularity(params.location, params.size);

	CUmemGenericAllocationHandle mem;
	if(auto err = cu.pcuMemCreate(&mem, params.granularSize, &prop, 0); CUDA_SUCCESS != err)
		return err;
	
	if (auto err = cu.pcuMemExportToShareableHandle(&params.osHandle, mem, prop.requestedHandleTypes, 0); CUDA_SUCCESS != err)
	{
		cu.pcuMemRelease(mem);
		return err;
	}

	if (auto err = reserveAdrressAndMapMemory(&params.ptr, params.granularSize, params.alignment, params.location, mem); CUDA_SUCCESS != err)
	{
		CloseHandle(params.osHandle);
		cu.pcuMemRelease(mem);
		return err;
	}

	if (auto err = cu.pcuMemRelease(mem); CUDA_SUCCESS != err)
	{
		CloseHandle(params.osHandle);
		return err;
	}
	
	*outMem = core::smart_refctd_ptr<CCUDASharedMemory>(new CCUDASharedMemory(core::smart_refctd_ptr<CCUDADevice>(this), std::move(params)), core::dont_grab);

	return CUDA_SUCCESS;
}

CUresult CCUDADevice::importGPUSemaphore(core::smart_refctd_ptr<CCUDASharedSemaphore>* outPtr, IGPUSemaphore* sema)
{
	if (!sema || !outPtr)
		return CUDA_ERROR_INVALID_VALUE;

	auto& cu = m_handler->getCUDAFunctionTable();
	auto handleType = sema->getCreationParams().externalHandleTypes.value;
	auto handle = sema->getCreationParams().externalHandle;

	if (!handleType || !handle)
		return CUDA_ERROR_INVALID_VALUE;

	CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC desc = {
		.type = static_cast<CUexternalSemaphoreHandleType>(handleType),
		.handle = {.win32 = {.handle = handle }},
	};

	CUexternalSemaphore cusema;
	if (auto err = cu.pcuImportExternalSemaphore(&cusema, &desc); CUDA_SUCCESS != err)
		return err;
	
	*outPtr = core::smart_refctd_ptr<CCUDASharedSemaphore>(new CCUDASharedSemaphore(core::smart_refctd_ptr<CCUDADevice>(this), core::smart_refctd_ptr<IGPUSemaphore>(sema), cusema, handle), core::dont_grab);
	return CUDA_SUCCESS;
}

#if 0
CUresult CCUDAHandler::registerBuffer(GraphicsAPIObjLink<video::IGPUBuffer>* link, uint32_t flags)
{
	assert(link->obj);
	auto glbuf = static_cast<video::COpenGLBuffer*>(link->obj.get());
	auto retval = cuda.pcuGraphicsGLRegisterBuffer(&link->cudaHandle,glbuf->getOpenGLName(),flags);
	if (retval!=CUDA_SUCCESS)
		link->obj = nullptr;
	return retval;
}
CUresult CCUDAHandler::registerImage(GraphicsAPIObjLink<video::IGPUImage>* link, uint32_t flags)
{
	assert(link->obj);
			
	auto format = link->obj->getCreationParameters().format;
	if (asset::isBlockCompressionFormat(format) || asset::isDepthOrStencilFormat(format) || asset::isScaledFormat(format) || asset::isPlanarFormat(format))
		return CUDA_ERROR_INVALID_IMAGE;

	auto glimg = static_cast<video::COpenGLImage*>(link->obj.get());
	GLenum target = glimg->getOpenGLTarget();
	switch (target)
	{
		case GL_TEXTURE_2D:
		case GL_TEXTURE_2D_ARRAY:
		case GL_TEXTURE_CUBE_MAP:
		case GL_TEXTURE_3D:
			break;
		default:
			return CUDA_ERROR_INVALID_IMAGE;
			break;
	}
	auto retval = cuda.pcuGraphicsGLRegisterImage(&link->cudaHandle,glimg->getOpenGLName(),target,flags);
	if (retval != CUDA_SUCCESS)
		link->obj = nullptr;
	return retval;
}


constexpr auto MaxAquireOps = 4096u;

CUresult CCUDAHandler::acquireAndGetPointers(GraphicsAPIObjLink<video::IGPUBuffer>* linksBegin, GraphicsAPIObjLink<video::IGPUBuffer>* linksEnd, CUstream stream, size_t* outbufferSizes)
{
	if (linksBegin+MaxAquireOps<linksEnd)
		return CUDA_ERROR_OUT_OF_MEMORY;
	alignas(_NBL_SIMD_ALIGNMENT) uint8_t stackScratch[MaxAquireOps*sizeof(void*)];

	CUresult result = acquireResourcesFromGraphics(stackScratch,linksBegin,linksEnd,stream);
	if (result != CUDA_SUCCESS)
		return result;

	size_t tmp = 0xdeadbeefbadc0ffeull;
	size_t* sit = outbufferSizes;
	for (auto iit=linksBegin; iit!=linksEnd; iit++,sit++)
	{
		if (!iit->acquired)
			return CUDA_ERROR_UNKNOWN;

		result = cuda::CCUDAHandler::cuda.pcuGraphicsResourceGetMappedPointer_v2(&iit->asBuffer.pointer,outbufferSizes ? sit:&tmp,iit->cudaHandle);
		if (result != CUDA_SUCCESS)
			return result;
	}
	return CUDA_SUCCESS;
}
CUresult CCUDAHandler::acquireAndGetMipmappedArray(GraphicsAPIObjLink<video::IGPUImage>* linksBegin, GraphicsAPIObjLink<video::IGPUImage>* linksEnd, CUstream stream)
{
	if (linksBegin+MaxAquireOps<linksEnd)
		return CUDA_ERROR_OUT_OF_MEMORY;
	alignas(_NBL_SIMD_ALIGNMENT) uint8_t stackScratch[MaxAquireOps*sizeof(void*)];

	CUresult result = acquireResourcesFromGraphics(stackScratch,linksBegin,linksEnd,stream);
	if (result != CUDA_SUCCESS)
		return result;

	for (auto iit=linksBegin; iit!=linksEnd; iit++)
	{
		if (!iit->acquired)
			return CUDA_ERROR_UNKNOWN;

		result = cuda::CCUDAHandler::cuda.pcuGraphicsResourceGetMappedMipmappedArray(&iit->asImage.mipmappedArray,iit->cudaHandle);
		if (result != CUDA_SUCCESS)
			return result;
	}
	return CUDA_SUCCESS;
}
CUresult CCUDAHandler::acquireAndGetArray(GraphicsAPIObjLink<video::IGPUImage>* linksBegin, GraphicsAPIObjLink<video::IGPUImage>* linksEnd, uint32_t* arrayIndices, uint32_t* mipLevels, CUstream stream)
{
	if (linksBegin+MaxAquireOps<linksEnd)
		return CUDA_ERROR_OUT_OF_MEMORY;
	alignas(_NBL_SIMD_ALIGNMENT) uint8_t stackScratch[MaxAquireOps*sizeof(void*)];

	CUresult result = acquireResourcesFromGraphics(stackScratch,linksBegin,linksEnd,stream);
	if (result != CUDA_SUCCESS)
		return result;

	auto ait = arrayIndices;
	auto mit = mipLevels;
	for (auto iit=linksBegin; iit!=linksEnd; iit++,ait++,mit++)
	{
		if (!iit->acquired)
			return CUDA_ERROR_UNKNOWN;

		result = cuda::CCUDAHandler::cuda.pcuGraphicsSubResourceGetMappedArray(&iit->asImage.array,iit->cudaHandle,*ait,*mit);
		if (result != CUDA_SUCCESS)
			return result;
	}
	return CUDA_SUCCESS;
}
#endif

}

#endif // _NBL_COMPILE_WITH_CUDA_
