#include "nbl/video/IGPUDescriptorSet.h"

namespace nbl::video
{

void IGPUDescriptorSet::allocateDescriptors()
{
    auto& offsets = m_descriptorStorageOffsets;

    for (uint32_t i = 0u; i < asset::EDT_COUNT; ++i)
    {
        const auto type = static_cast<asset::E_DESCRIPTOR_TYPE>(i);
        const auto count = getLayout()->getTotalDescriptorCount(type);
        if (count == 0ull)
            continue;

        if (m_pool->m_flags & IDescriptorPool::ECF_FREE_DESCRIPTOR_SET_BIT)
            offsets.data[type] = m_pool->m_generalAllocators[type].alloc_addr(count, 1u);
        else
            offsets.data[type] = m_pool->m_linearAllocators[type].alloc_addr(count, 1u);

        assert((offsets.data[type] < m_pool->m_maxDescriptorCount[i]) && "PANIC: Allocation failed. This shoudn't have happened! Check your descriptor pool.");
    }

    const auto mutableSamplerCount = getLayout()->getTotalMutableSamplerCount();
    if (mutableSamplerCount != 0ull)
    {
        if (m_pool->m_flags & IDescriptorPool::ECF_FREE_DESCRIPTOR_SET_BIT)
            offsets.data[asset::EDT_COUNT] = m_pool->m_generalAllocators[asset::EDT_COUNT].alloc_addr(mutableSamplerCount, 1u);
        else
            offsets.data[asset::EDT_COUNT] = m_pool->m_linearAllocators[asset::EDT_COUNT].alloc_addr(mutableSamplerCount, 1u);
    }
}

}