#include "nbl/video/IDescriptorPool.h"
#include "nbl/video/IGPUDescriptorSetLayout.h"

namespace nbl::video
{

bool IDescriptorPool::createDescriptorSets(uint32_t count, const IGPUDescriptorSetLayout* const* layouts, core::smart_refctd_ptr<IGPUDescriptorSet>* output)
{
    core::vector<SDescriptorOffsets> descriptorOffsets(count);

    for (uint32_t i = 0u; i < count; ++i)
    {
        if (!isCompatibleDevicewise(layouts[i]))
            return false;
        
        descriptorOffsets[i] = allocateDescriptorOffsets(layouts[i]);
    }

    return createDescriptorSets_impl(count, layouts, descriptorOffsets.data(), output);
}

IDescriptorPool::SDescriptorOffsets IDescriptorPool::allocateDescriptorOffsets(const IGPUDescriptorSetLayout* layout)
{
    SDescriptorOffsets offsets;

    for (uint32_t i = 0u; i < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++i)
    {
        const auto type = static_cast<asset::IDescriptor::E_TYPE>(i);
        const auto count = layout->getTotalDescriptorCount(type);
        if (count == 0ull)
            continue;

        if (m_flags & IDescriptorPool::ECF_FREE_DESCRIPTOR_SET_BIT)
            offsets.data[i] = m_generalAllocators[i].alloc_addr(count, 1u);
        else
            offsets.data[i] = m_linearAllocators[i].alloc_addr(count, 1u);

        assert((offsets.data[i] < m_maxDescriptorCount[i]) && "PANIC: Allocation failed. This shoudn't have happened! Check your descriptor pool.");
    }

    const auto mutableSamplerCount = layout->getTotalMutableSamplerCount();
    if (mutableSamplerCount != 0ull)
    {
        if (m_flags & IDescriptorPool::ECF_FREE_DESCRIPTOR_SET_BIT)
            offsets.data[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)] = m_generalAllocators[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)].alloc_addr(mutableSamplerCount, 1u);
        else
            offsets.data[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)] = m_linearAllocators[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)].alloc_addr(mutableSamplerCount, 1u);
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
        for (auto t = 0u; t < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++i)
        {
            const auto type = static_cast<asset::IDescriptor::E_TYPE>(t);

            const uint32_t allocatedOffset = descriptorSets[i]->getDescriptorStorageOffset(type);
            if (allocatedOffset == ~0u)
                continue;

            const uint32_t count = descriptorSets[i]->getLayout()->getTotalDescriptorCount(type);
            assert(count != 0u);

            auto* descriptors = descriptorSets[i]->getAllDescriptors(type);
            assert(descriptors);

            for (auto c = 0u; c < count; ++c)
                descriptors[c].~smart_refctd_ptr();

            m_generalAllocators[t].free_addr(allocatedOffset, count);
        }

        const uint32_t count = descriptorSets[i]->getLayout()->getTotalMutableSamplerCount();
        if (count > 0)
        {
            const uint32_t allocatedOffset = descriptorSets[i]->getMutableSamplerStorageOffset();
            if (allocatedOffset == ~0u)
                continue;

            auto* samplers = descriptorSets[i]->getAllMutableSamplers();
            assert(samplers);

            for (auto c = 0u; c < count; ++c)
                samplers[c].~smart_refctd_ptr();

            m_generalAllocators[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)].free_addr(allocatedOffset, count);
        }
    }

    return freeDescriptorSets_impl(descriptorSetCount, descriptorSets);
}

}