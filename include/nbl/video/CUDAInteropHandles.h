// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_CUDA_INTEROP_HANDLES_H_INCLUDED_
#define _NBL_VIDEO_CUDA_INTEROP_HANDLES_H_INCLUDED_

#include <cstddef>
#include <cstdint>

namespace nbl::video::cuda_interop
{

struct alignas(alignof(int32_t)) SCUdevice
{
	uint8_t value[sizeof(int32_t)] = {};
};

struct alignas(alignof(void*)) SCUcontext
{
	uint8_t value[sizeof(void*)] = {};
};

struct alignas(alignof(uintptr_t)) SCUdeviceptr
{
	uint8_t value[sizeof(uintptr_t)] = {};
};

struct alignas(alignof(void*)) SCUexternalMemory
{
	uint8_t value[sizeof(void*)] = {};
};

struct alignas(alignof(void*)) SCUexternalSemaphore
{
	uint8_t value[sizeof(void*)] = {};
};

}

#endif
