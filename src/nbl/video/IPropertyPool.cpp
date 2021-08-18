#include "nbl/video/ILogicalDevice.h"

using namespace nbl;
using namespace video;


#define PROPERTY_ADDRESS_ALLOCATOR_ARGS 1u,capacity,1u

IPropertyPool::PropertyAddressAllocator::size_type IPropertyPool::getReservedSize(uint32_t capacity, bool contiguous)
{
    auto size = PropertyAddressAllocator::reserved_size(PROPERTY_ADDRESS_ALLOCATOR_ARGS);
    if (contiguous)
        size += 2u*capacity*sizeof(uint32_t);
    return size;
}

IPropertyPool::IPropertyPool(uint32_t capacity, void* reserved, bool contiguous) : indexAllocator(reserved,0u,0u,PROPERTY_ADDRESS_ALLOCATOR_ARGS), m_indexToAddr(nullptr), m_addrToIndex(nullptr)
{
    if (contiguous)
    {
        m_indexToAddr = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(reserved)+getReservedSize(capacity));
        m_addrToIndex = m_indexToAddr+capacity;

        std::fill_n(m_indexToAddr,capacity,invalid_index);
        std::fill_n(m_addrToIndex,capacity,invalid_index);
    }
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