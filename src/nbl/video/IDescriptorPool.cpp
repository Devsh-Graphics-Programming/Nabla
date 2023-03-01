#include "nbl/video/IDescriptorPool.h"
#include "nbl/video/IGPUDescriptorSetLayout.h"

namespace nbl::video
{

uint32_t IDescriptorPool::createDescriptorSets(uint32_t count, const IGPUDescriptorSetLayout* const* layouts, core::smart_refctd_ptr<IGPUDescriptorSet>* output)
{
    core::vector<SDescriptorOffsets> descriptorOffsets;
    descriptorOffsets.reserve(count);

    system::ILogger* logger = nullptr;
    {
        auto debugCallback = getOriginDevice()->getPhysicalDevice()->getDebugCallback();
        if (debugCallback)
            logger = debugCallback->getLogger();
    }

    for (uint32_t i = 0u; i < count; ++i)
    {
        if (!isCompatibleDevicewise(layouts[i]))
        {
            if (logger)
                logger->log("Device-Incompatible descriptor set layout found at index %u. Sets for the layouts following this index will not be created.", system::ILogger::ELL_WARNING, i);

            break;
        }
        
        if (!allocateDescriptorOffsets(descriptorOffsets.emplace_back(), layouts[i]))
        {
            if (logger)
                logger->log("Failed to allocate descriptor offsets in the pool's storage for descriptor set layout at index %u. Sets for the layouts following this index will not be created.", system::ILogger::ELL_WARNING, i);

            descriptorOffsets.pop_back();
            break;
        }
    }

    const auto successCount = descriptorOffsets.size();
    assert(count >= successCount);
    std::fill_n(output + successCount, count - successCount, nullptr);

    const uint32_t allocatedDSOffset = m_descriptorSetAllocator.alloc_addr(successCount, 1u);
    if (allocatedDSOffset != m_descriptorSetAllocator.invalid_address)
    {
        if (createDescriptorSets_impl(successCount, layouts, descriptorOffsets.data(), allocatedDSOffset, output))
        {
            for (uint32_t i = 0u; i < successCount; ++i)
                m_allocatedDescriptorSets[allocatedDSOffset + i] = output[i].get();

            return successCount;
        }
    }
    else
    {
        if (logger)
            logger->log("Failed to allocate descriptor sets.", system::ILogger::ELL_ERROR);
    }

    // Free the allocated offsets for all the successfully allocated descriptor sets.
    for (uint32_t i = 0u; i < successCount; ++i)
        freeDescriptorOffsets(descriptorOffsets[i], layouts[i]);

    return 0;
}

bool IDescriptorPool::reset()
{
    for (const uint32_t setIndex : m_descriptorSetAllocator)
    {
        deleteSetStorage(m_allocatedDescriptorSets[setIndex]);

        m_allocatedDescriptorSets[setIndex]->m_pool = nullptr;
        m_allocatedDescriptorSets[setIndex]->m_poolOffset = ~0u;
        m_allocatedDescriptorSets[setIndex] = nullptr;
    }

    m_descriptorSetAllocator.reset();

    return reset_impl();
}

bool IDescriptorPool::allocateDescriptorOffsets(SDescriptorOffsets& offsets, const IGPUDescriptorSetLayout* layout)
{
    for (uint32_t i = 0u; i <= static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++i)
    {
        uint32_t count = 0u;
        uint32_t maxCount = 0u;
        if (i == static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT))
        {
            count = layout->getTotalMutableSamplerCount();
            maxCount = m_creationParameters.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER)];
        }
        else
        {
            const auto type = static_cast<asset::IDescriptor::E_TYPE>(i);
            count = layout->getTotalDescriptorCount(type);
            maxCount = m_creationParameters.maxDescriptorCount[i];
        }

        if (count == 0u)
            continue;

        offsets.data[i] = m_descriptorAllocators[i]->allocate(count);
        if (offsets.data[i] >= maxCount)
        {
            // Offset allocation for this descriptor type failed, nothing needs to be done for this type but rewind the previous types' allocations.
            freeDescriptorOffsets(offsets, layout);
            return false;
        }
    }

    return true;
}

void IDescriptorPool::freeDescriptorOffsets(SDescriptorOffsets& offsets, const IGPUDescriptorSetLayout* layout)
{
    for (uint32_t i = 0; i <= static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++i)
    {
        const uint32_t maxCount = (i == static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT))
            ? m_creationParameters.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER)]
            : m_creationParameters.maxDescriptorCount[i];

        if (offsets.data[i] < maxCount)
        {
            if (allowsFreeing())
                m_descriptorAllocators[i]->free(offsets.data[i], layout->getTotalDescriptorCount(static_cast<asset::IDescriptor::E_TYPE>(i)));
            else
                m_descriptorAllocators[i]->linearAllocator.reset(offsets.data[i]);
        }
    }
}

void IDescriptorPool::deleteSetStorage(IGPUDescriptorSet* set)
{
    assert(set);

    for (auto i = 0u; i < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++i)
    {
        const auto type = static_cast<asset::IDescriptor::E_TYPE>(i);
        const uint32_t allocatedOffset = set->getDescriptorStorageOffset(type);

        // There is no descriptor of such type in the set.
        if (allocatedOffset == ~0u)
            continue;

        const uint32_t count = set->getLayout()->getTotalDescriptorCount(type);
        assert(count != 0u);

        std::destroy_n(getDescriptorStorage(type) + allocatedOffset, count);

        if (allowsFreeing())
            m_descriptorAllocators[i]->free(allocatedOffset, count);
    }

    const auto mutableSamplerCount = set->getLayout()->getTotalMutableSamplerCount();
    if (mutableSamplerCount > 0)
    {
        const uint32_t allocatedOffset = set->getMutableSamplerStorageOffset();
        assert(allocatedOffset != ~0u);

        std::destroy_n(getMutableSamplerStorage() + allocatedOffset, mutableSamplerCount);

        if (allowsFreeing())
            m_descriptorAllocators[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)]->free(allocatedOffset, mutableSamplerCount);
    } 
}

}