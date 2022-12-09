#include "nbl/video/IDescriptorPool.h"
#include "nbl/video/IGPUDescriptorSetLayout.h"

namespace nbl::video
{

core::smart_refctd_ptr<IGPUDescriptorSet> IDescriptorPool::createDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& layout)
{
    if (!isCompatibleDevicewise(layout.get()))
        return nullptr;

    return createDescriptorSet_impl(std::move(layout));
}

bool IDescriptorPool::updateDescriptorSets(uint32_t descriptorWriteCount, const IGPUDescriptorSet::SWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const IGPUDescriptorSet::SCopyDescriptorSet* pDescriptorCopies)
{
    for (auto i = 0; i < descriptorWriteCount; ++i)
    {
        auto* ds = static_cast<IGPUDescriptorSet*>(pDescriptorWrites[i].dstSet);
        ds->incrementVersion();

        auto* descriptors = ds->getDescriptors(pDescriptorWrites[i].descriptorType, pDescriptorWrites[i].binding);
        auto* samplers = ds->getMutableSamplers(pDescriptorWrites[i].binding);
        for (auto j = 0; j < pDescriptorWrites[i].count; ++j)
        {
            descriptors[j] = pDescriptorWrites[i].info[j].desc;

            if (samplers)
                samplers[j] = pDescriptorWrites[i].info[j].info.image.sampler;
        }
    }

    for (auto i = 0; i < descriptorCopyCount; ++i)
    {
        const auto* srcDS = static_cast<const IGPUDescriptorSet*>(pDescriptorCopies[i].srcSet);
        auto* dstDS = static_cast<IGPUDescriptorSet*>(pDescriptorCopies[i].dstSet);

        auto foundBindingInfo = std::lower_bound(srcDS->getLayout()->getBindings().begin(), srcDS->getLayout()->getBindings().end(), pDescriptorCopies[i].srcBinding,
            [](const IGPUDescriptorSetLayout::SBinding& a, const uint32_t b) -> bool
            {
                return a.binding < b;
            });

        if (foundBindingInfo->binding != pDescriptorCopies[i].srcBinding)
            return false;

        const asset::E_DESCRIPTOR_TYPE descriptorType = foundBindingInfo->type;

        auto* srcDescriptors = srcDS->getDescriptors(descriptorType, pDescriptorCopies[i].srcBinding);
        auto* srcSamplers = srcDS->getMutableSamplers(pDescriptorCopies[i].srcBinding);
        if (!srcDescriptors)
            return false;

        auto* dstDescriptors = dstDS->getDescriptors(descriptorType, pDescriptorCopies[i].dstBinding);
        auto* dstSamplers = dstDS->getMutableSamplers(pDescriptorCopies[i].dstBinding);
        if (!dstDescriptors)
            return false;

        memcpy(dstDescriptors, srcDescriptors, pDescriptorCopies[i].count * sizeof(core::smart_refctd_ptr<const asset::IDescriptor>));

        if (srcSamplers && dstSamplers)
            memcpy(dstSamplers, srcSamplers, pDescriptorCopies[i].count * sizeof(core::smart_refctd_ptr<const IGPUSampler>));
    }

    updateDescriptorSets_impl(descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies);

    return true;
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

            auto* descriptors = descriptorSets[i]->getAllDescriptors(type);
            assert(descriptors);

            for (auto c = 0u; c < count; ++c)
                descriptors[c].~smart_refctd_ptr();

            m_generalAllocators[type].free_addr(allocatedOffset, count);
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

            m_generalAllocators[asset::EDT_COUNT].free_addr(allocatedOffset, count);
        }
    }

    return freeDescriptorSets_impl(descriptorSetCount, descriptorSets);
}

}