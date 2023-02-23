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

bool IDescriptorPool::reset()
{
    const bool allowsFreeing = m_creationParameters.flags.hasFlags(ECF_FREE_DESCRIPTOR_SET_BIT);
    for (uint32_t t = 0; t < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++t)
    {
        if (!m_descriptorAllocators[t])
            continue;

        // TODO(achal): I should call the dtors on descriptors only after calling vkResetDescriptorPool (in reset_impl)
        const uint32_t allocatedCount = m_descriptorAllocators[t]->getAllocatedDescriptorCount(allowsFreeing);
        std::destroy_n(getDescriptorStorage(static_cast<asset::IDescriptor::E_TYPE>(t)), allocatedCount);

        m_descriptorAllocators[t]->reset(allowsFreeing);
    }

    m_version.fetch_add(1u);

    return reset_impl();
}

IDescriptorPool::SDescriptorOffsets IDescriptorPool::allocateDescriptorOffsets(const IGPUDescriptorSetLayout* layout)
{
    SDescriptorOffsets offsets;

    const bool allowsFreeing = m_creationParameters.flags.hasFlags(ECF_FREE_DESCRIPTOR_SET_BIT);

    for (uint32_t i = 0u; i < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++i)
    {
        const auto type = static_cast<asset::IDescriptor::E_TYPE>(i);
        const auto count = layout->getTotalDescriptorCount(type);
        if (count == 0ull)
            continue;

        offsets.data[i] = m_descriptorAllocators[i]->allocate(count, allowsFreeing);

        assert((offsets.data[i] < m_creationParameters.maxDescriptorCount[i]) && "PANIC: Allocation failed. This shoudn't have happened! Check your descriptor pool.");
    }

    const auto mutableSamplerCount = layout->getTotalMutableSamplerCount();
    if (mutableSamplerCount != 0ull)
        offsets.data[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)] = m_descriptorAllocators[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)]->allocate(mutableSamplerCount, allowsFreeing);

    return offsets;
}

}