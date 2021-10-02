// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/alloc/SimpleGPUBufferAllocator.h"


using namespace nbl;
using namespace video;


SimpleGPUBufferAllocator::value_type SimpleGPUBufferAllocator::allocate(size_t bytes, size_t alignment) noexcept
{
    auto reqs = mBufferMemReqs;
    reqs.vulkanReqs.size = bytes;
    reqs.vulkanReqs.alignment = alignment;

    // Todo(achal): Need to get these values from the user somehow
    // This doesn't matter right now because updateBufferRangeViaStagingBuffer 
    // only makes use of upstreaming buffer which would be TRANSFER_SRC
    IGPUBuffer::SCreationParams creationParams = {};
    creationParams.usage = IGPUBuffer::EUF_TRANSFER_SRC_BIT;
    creationParams.sharingMode = asset::ESM_EXCLUSIVE;
    creationParams.queueFamilyIndexCount = 0u;
    creationParams.queueFamilyIndices = nullptr;

    return mDriver->createGPUBufferOnDedMem(creationParams, reqs, canUpdateViaCmdBuff);
}
