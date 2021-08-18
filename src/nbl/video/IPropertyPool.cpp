#include "nbl/video/ILogicalDevice.h"

using namespace nbl;
using namespace video;


#define PROPERTY_ADDRESS_ALLOCATOR_ARGS 1u,capacity,1u

IPropertyPool::PropertyAddressAllocator::size_type IPropertyPool::getReservedSize(uint32_t capacity)
{
    return PropertyAddressAllocator::reserved_size(PROPERTY_ADDRESS_ALLOCATOR_ARGS);
}

IPropertyPool::IPropertyPool(uint32_t capacity, void* reserved) : indexAllocator(reserved,0u,0u,PROPERTY_ADDRESS_ALLOCATOR_ARGS)
{
}

bool IPropertyPool::validateBlocks(const ILogicalDevice* device, const IPropertyPool& declvalPool, const uint32_t capacity, const asset::SBufferRange<IGPUBuffer>* _memoryBlocks)
{
    for (auto i=0u; i<declvalPool.getPropertyCount(); i++)
    {
        const auto& memBlk = _memoryBlocks[i];
        if (!memBlk.isValid())
            return false;
        if (memBlk.offset%device->getPhysicalDevice()->getLimits().SSBOAlignment)
            return false;
        if (memBlk.size<declvalPool.getPropertySize(i)*capacity)
            return false;
    }
    return true;
}