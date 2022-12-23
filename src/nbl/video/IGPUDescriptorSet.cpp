#include "nbl/video/IGPUDescriptorSet.h"

#include "nbl/video/IDescriptorPool.h"

namespace nbl::video
{

IGPUDescriptorSet::IGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& layout, core::smart_refctd_ptr<IDescriptorPool>&& pool, IDescriptorPool::SDescriptorOffsets&& offsets)
    : base_t(std::move(layout)), IBackendObject(std::move(core::smart_refctd_ptr<const ILogicalDevice>(pool->getOriginDevice()))), m_version(0ull), m_parentPoolVersion(pool->m_version), m_pool(std::move(pool)), m_descriptorStorageOffsets(std::move(offsets))
{
    for (auto i = 0u; i < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++i)
    {
        // There is no descriptor of such type in the set.
        if (m_descriptorStorageOffsets.data[i] == ~0u)
            continue;

        const auto type = static_cast<asset::IDescriptor::E_TYPE>(i);

        // Default-construct the core::smart_refctd_ptr<IDescriptor>s because even if the user didn't update the descriptor set with ILogicalDevice::updateDescriptorSet we
        // won't have uninitialized memory and destruction wouldn't crash in ~IGPUDescriptorSet.
        std::uninitialized_default_construct_n(m_pool->getDescriptorStorage(type) + m_descriptorStorageOffsets.data[i], m_layout->getTotalDescriptorCount(type));
    }

    const auto mutableSamplerCount = m_layout->getTotalMutableSamplerCount();
    if (mutableSamplerCount > 0)
        std::uninitialized_default_construct_n(m_pool->getMutableSamplerStorage() + m_descriptorStorageOffsets.data[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)], mutableSamplerCount);
}

IGPUDescriptorSet::~IGPUDescriptorSet()
{
    if (!isZombie())
    {
        const bool allowsFreeing = m_pool->m_flags & IDescriptorPool::ECF_FREE_DESCRIPTOR_SET_BIT;
        for (auto i = 0u; i < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++i)
        {
            // There is no descriptor of such type in the set.
            if (m_descriptorStorageOffsets.data[i] == ~0u)
                continue;

            const auto type = static_cast<asset::IDescriptor::E_TYPE>(i);

            const uint32_t allocatedOffset = getDescriptorStorageOffset(type);
            assert(allocatedOffset != ~0u);

            const uint32_t count = m_layout->getTotalDescriptorCount(type);
            assert(count != 0u);

            std::destroy_n(m_pool->getDescriptorStorage(type) + allocatedOffset, count);

            if (allowsFreeing)
                m_pool->m_descriptorAllocators[i]->free(allocatedOffset, count);
        }

        const auto mutableSamplerCount = m_layout->getTotalMutableSamplerCount();
        if (mutableSamplerCount > 0)
        {
            const uint32_t allocatedOffset = getMutableSamplerStorageOffset();
            assert(allocatedOffset != ~0u);

            std::destroy_n(m_pool->getMutableSamplerStorage() + allocatedOffset, mutableSamplerCount);

            if (allowsFreeing)
                m_pool->m_descriptorAllocators[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)]->free(allocatedOffset, mutableSamplerCount);
        }
    }
}

}