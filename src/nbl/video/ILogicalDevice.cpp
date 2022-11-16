#include "nbl/video/IPhysicalDevice.h"

using namespace nbl;
using namespace nbl::video;


E_API_TYPE ILogicalDevice::getAPIType() const { return m_physicalDevice->getAPIType(); }

core::smart_refctd_ptr<IGPUDescriptorSetLayout> ILogicalDevice::createDescriptorSetLayout(const IGPUDescriptorSetLayout::SBinding* _begin, const IGPUDescriptorSetLayout::SBinding* _end)
{
    uint32_t dynamicSSBOCount=0u,dynamicUBOCount=0u;
    for (auto b=_begin; b!=_end; ++b)
    {
        if (b->type == asset::EDT_STORAGE_BUFFER_DYNAMIC)
            dynamicSSBOCount++;
        else if (b->type == asset::EDT_UNIFORM_BUFFER_DYNAMIC)
            dynamicUBOCount++;
        else if (b->type == asset::EDT_COMBINED_IMAGE_SAMPLER && b->samplers)
        {
            auto* samplers = b->samplers;
            for (uint32_t i = 0u; i < b->count; ++i)
                if (!samplers[i]->wasCreatedBy(this))
                    return nullptr;
        }
    }
    const auto& limits = m_physicalDevice->getLimits();
    if (dynamicSSBOCount>limits.maxDescriptorSetDynamicOffsetSSBOs || dynamicUBOCount>limits.maxDescriptorSetDynamicOffsetUBOs)
        return nullptr;
    return createDescriptorSetLayout_impl(_begin,_end);
}

bool ILogicalDevice::updateDescriptorSets(uint32_t descriptorWriteCount, const IGPUDescriptorSet::SWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const IGPUDescriptorSet::SCopyDescriptorSet* pDescriptorCopies)
{
    for (auto i = 0; i < descriptorWriteCount; ++i)
    {
        auto* ds = static_cast<IGPUDescriptorSet*>(pDescriptorWrites[i].dstSet);

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

        // This memcpy will increment the reference count, won't it?
        memcpy(dstDescriptors, srcDescriptors, pDescriptorCopies[i].count * sizeof(core::smart_refctd_ptr<const asset::IDescriptor>));

        if (srcSamplers && dstSamplers)
            memcpy(dstSamplers, srcSamplers, pDescriptorCopies[i].count * sizeof(core::smart_refctd_ptr<const IGPUSampler>));
    }

    updateDescriptorSets_impl(descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies);

    return true;
}
