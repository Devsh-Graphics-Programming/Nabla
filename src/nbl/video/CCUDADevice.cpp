// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/video/CCUDADevice.h"

#ifdef _NBL_COMPILE_WITH_CUDA_
namespace nbl::video
{

CCUDADevice::CCUDADevice(core::smart_refctd_ptr<CVulkanConnection>&& _vulkanConnection, IPhysicalDevice* const _vulkanDevice, const E_VIRTUAL_ARCHITECTURE _virtualArchitecture, CUdevice _handle, core::smart_refctd_ptr<CCUDAHandler>&& _handler)
	: m_defaultCompileOptions(), m_vulkanConnection(std::move(_vulkanConnection)), m_vulkanDevice(_vulkanDevice), m_virtualArchitecture(_virtualArchitecture), m_handle(_handle), m_handler(std::move(_handler))
{
	m_defaultCompileOptions.push_back("--std=c++14");
	m_defaultCompileOptions.push_back(virtualArchCompileOption[m_virtualArchitecture]);
	m_defaultCompileOptions.push_back("-dc");
	m_defaultCompileOptions.push_back("-use_fast_math");
	m_handler->getCUDAFunctionTable().pcuCtxCreate_v2(&m_context, 0, m_handle);
}

CCUDADevice::~CCUDADevice()
{
	m_handler->getCUDAFunctionTable().pcuCtxDestroy_v2(m_context);
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
