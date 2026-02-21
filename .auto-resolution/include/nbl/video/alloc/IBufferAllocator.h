// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_BUFFER_ALLOCATOR_BASE_H_
#define _NBL_VIDEO_I_BUFFER_ALLOCATOR_BASE_H_

#include "nbl/video/IGPUBuffer.h"

namespace nbl::video
{

class IBufferAllocator
{
    protected:
        IBufferAllocator() = default;
        virtual ~IBufferAllocator() = default;
};

}
#endif
