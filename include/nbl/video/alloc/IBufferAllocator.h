// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_BUFFER_ALLOCATOR_BASE_H__
#define __NBL_VIDEO_I_BUFFER_ALLOCATOR_BASE_H__

#include "nbl/video/IDriverMemoryAllocator.h"

namespace nbl::video
{

class IBufferAllocator
{
    protected:
        core::smart_refctd_ptr<IDriverMemoryAllocator> m_memoryAllocator;

        IBufferAllocator(core::smart_refctd_ptr<IDriverMemoryAllocator>&& _memoryAllocator) : m_memoryAllocator(std::move(_memoryAllocator)) {}
        virtual ~IBufferAllocator() = default;
};

}


#endif
