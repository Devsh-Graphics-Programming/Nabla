#include "nbl/video/IGPUDescriptorSet.h"
#include "nbl/video/IDescriptorPool.h"

namespace nbl::video
{

uint8_t* IGPUDescriptorSet::getDescriptorMemory(const asset::E_DESCRIPTOR_TYPE type, const uint32_t binding) const
{
    assert((m_descriptorStorageOffsets.data[type] != ~0u) && "The parent pool doesn't allow for this descriptor!");

    auto* baseAddress = getDescriptorStorage(type);
    if (baseAddress == nullptr)
        return nullptr;

    const uint32_t localOffset = m_layout->getDescriptorOffsetForBinding(binding);
    if (localOffset == ~0u)
        return nullptr;

    return reinterpret_cast<uint8_t*>(baseAddress + m_descriptorStorageOffsets.data[type] + localOffset);
}

}