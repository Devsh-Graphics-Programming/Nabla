#include "nbl/video/IGPUDescriptorSet.h"
#include "nbl/video/IDescriptorPool.h"

namespace nbl::video
{

uint8_t* IGPUDescriptorSet::getDescriptorMemory(const asset::E_DESCRIPTOR_TYPE type) const
{
    assert((m_descriptorStorageOffsets[type] != ~0u) && "The parent pool doesn't allow for this descriptor!");
    auto* baseAddress = m_pool->getDescriptorMemoryBaseAddress(type);
    if (baseAddress == nullptr)
        return nullptr;

    return baseAddress + m_descriptorStorageOffsets[type];
}

}