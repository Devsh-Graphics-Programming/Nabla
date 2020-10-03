// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "irr/video/alloc/GPUMemoryAllocatorBase.h"

#include "IVideoDriver.h"

using namespace irr;
using namespace video;


void            GPUMemoryAllocatorBase::copyBuffersWrapper(IGPUBuffer* oldBuffer, IGPUBuffer* newBuffer, size_t oldOffset, size_t newOffset, size_t copyRangeLen)
{
    mDriver->copyBuffer(oldBuffer,newBuffer,oldOffset,newOffset,copyRangeLen);
}
