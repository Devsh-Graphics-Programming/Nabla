#include "nbl/video/IGPUDescriptorSet.h"
#include "nbl/video/IDescriptorPool.h"

namespace nbl::video
{

uint8_t* IGPUDescriptorSet::getDescriptorMemory(const asset::E_DESCRIPTOR_TYPE type, const uint32_t binding) const
{
    assert((m_descriptorStorageOffsets.data[type] != ~0u) && "The parent pool doesn't allow for this descriptor!");

    auto* baseAddress = m_pool->getDescriptorMemoryBaseAddress(type);
    if (baseAddress == nullptr)
        return nullptr;

    const uint32_t localOffset = m_layout->getDescriptorOffsetForBinding(binding);
    if (localOffset == ~0u)
        return nullptr;

    return reinterpret_cast<uint8_t*>(reinterpret_cast<core::smart_refctd_ptr<const asset::IDescriptor>*>(baseAddress) + m_descriptorStorageOffsets.data[type] + localOffset);
}

}