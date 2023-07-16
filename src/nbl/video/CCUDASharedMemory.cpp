// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/video/CCUDASharedMemory.h"

#ifdef _NBL_COMPILE_WITH_CUDA_
namespace nbl::video
{
CCUDASharedMemory::~CCUDASharedMemory()
{
	auto& cu = m_device->getHandler()->getCUDAFunctionTable();
	auto& params = m_params;
	if (params.srcRes)
	{
		cu.pcuDestroyExternalMemory(params.extMem); 
		params.srcRes->drop();
	}
	else
	{
		cu.pcuMemUnmap(params.ptr, params.size); 
		cu.pcuMemAddressFree(params.ptr, params.size);
		cu.pcuMemRelease(params.mem); 
	}
	CloseHandle(params.osHandle);
}
}

#endif // _NBL_COMPILE_WITH_CUDA_
