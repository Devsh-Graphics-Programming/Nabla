#include "nbl/video/IDescriptorPool.h"
#include "nbl/video/IGPUDescriptorSetLayout.h"

namespace nbl::video
{

IDescriptorPool::SDescriptorOffsets IDescriptorPool::allocateDescriptors(const IGPUDescriptorSetLayout* layout)
{
    SDescriptorOffsets offsets;

    for (uint32_t i = 0u; i < asset::EDT_COUNT; ++i)
    {
        const auto type = static_cast<asset::E_DESCRIPTOR_TYPE>(i);
        const auto count = layout->getTotalDescriptorCount(type);
        if (count == 0ull)
            continue;

        if (m_flags & ECF_FREE_DESCRIPTOR_SET_BIT)
            offsets.data[type] = m_generalAllocators[type].alloc_addr(count, 1u);
        else
            offsets.data[type] = m_linearAllocators[type].alloc_addr(count, 1u);

        assert((offsets.data[type] < m_maxDescriptorCount[i]) && "PANIC: Allocation failed. This shoudn't have happened!");
    }

    return offsets;
}

bool IDescriptorPool::freeDescriptorSets(const uint32_t descriptorSetCount, IGPUDescriptorSet* const* const descriptorSets)
{
    const bool allowsFreeingDescriptorSets = m_flags & IDescriptorPool::ECF_FREE_DESCRIPTOR_SET_BIT;
    if (!allowsFreeingDescriptorSets)
        return false;

    for (auto i = 0u; i < descriptorSetCount; ++i)
    {
        for (auto t = 0u; t < asset::EDT_COUNT; ++i)
        {
            const auto type = static_cast<asset::E_DESCRIPTOR_TYPE>(t);

            const uint32_t allocatedOffset = descriptorSets[i]->getDescriptorStorageOffset(type);
            if (allocatedOffset == ~0u)
                continue;

            const uint32_t count = descriptorSets[i]->getLayout()->getTotalDescriptorCount(type);
            assert(count != 0u);

            auto* descriptors = descriptorSets[i]->getDescriptors(type);
            assert(descriptors);

            for (auto c = 0u; c < count; ++c)
                descriptors[c].~smart_refctd_ptr();

            m_generalAllocators[type].free_addr(allocatedOffset, count);
        }
    }

    return freeDescriptorSets_impl(descriptorSetCount, descriptorSets);
}

}