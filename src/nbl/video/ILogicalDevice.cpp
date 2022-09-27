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
    // TODO(achal): Allow for this behaviour from the spec:
    // > If the dstBinding has fewer than descriptorCount array elements remaining starting from dstArrayElement,
    // > then the remainder will be used to update the subsequent binding - dstBinding+1 starting at array element zero.
    // >
    // > If a binding has a descriptorCount of zero, it is skipped.
    // >
    // > This behavior applies recursively, with the update affecting consecutive bindings as needed to update all descriptorCount descriptors.
    // >
    // > Consecutive bindings must have identical VkDescriptorType, VkShaderStageFlags, VkDescriptorBindingFlagBits, and immutable samplers references.

    auto getDescriptorMemoryForBinding = [](const uint32_t binding, const asset::E_DESCRIPTOR_TYPE type, const IGPUDescriptorSet* set) -> void*
    {
        const uint32_t localOffset = set->getLayout()->getDescriptorOffsetForBinding(binding);

        auto* baseAddress = set->getDescriptorMemory(type);
        if (!baseAddress)
            return nullptr;

        return reinterpret_cast<void*>(baseAddress + localOffset);
    };

    for (auto i = 0; i < descriptorWriteCount; ++i)
    {
        auto* ds = static_cast<IGPUDescriptorSet*>(pDescriptorWrites[i].dstSet);
        void* descriptorMemory = getDescriptorMemoryForBinding(pDescriptorWrites[i].binding, pDescriptorWrites[i].descriptorType, ds);

        auto* descriptors = new (descriptorMemory) core::smart_refctd_ptr<const asset::IDescriptor>[pDescriptorWrites[i].count];

        for (auto j = 0; j < pDescriptorWrites[i].count; ++j)
            descriptors[j] = pDescriptorWrites[i].info[j].desc;
    }

    for (auto i = 0; i < descriptorCopyCount; ++i)
    {
        const auto* srcDS = static_cast<const IGPUDescriptorSet*>(pDescriptorCopies[i].srcSet);
        auto* dstDS = static_cast<IGPUDescriptorSet*>(pDescriptorCopies[i].dstSet);

        asset::E_DESCRIPTOR_TYPE descriptorType = asset::EDT_COUNT;
        for (const auto& binding : srcDS->getLayout()->getBindings())
        {
            if (binding.binding == pDescriptorCopies[i].srcBinding)
            {
                descriptorType = binding.type;

                auto* srcDescriptors = reinterpret_cast<core::smart_refctd_ptr<const asset::IDescriptor>*>(getDescriptorMemoryForBinding(pDescriptorCopies[i].srcBinding, descriptorType, srcDS));
                if (!srcDescriptors)
                    return false;

                auto* dstDescriptorMemory = getDescriptorMemoryForBinding(pDescriptorCopies[i].dstBinding, descriptorType, dstDS);
                if (!dstDescriptorMemory)
                    return false;

                auto* dstDescriptors = new (dstDescriptorMemory) core::smart_refctd_ptr<const asset::IDescriptor>[pDescriptorCopies[i].count];

                // This memcpy will increment the reference count, won't it?
                memcpy(dstDescriptors, srcDescriptors, pDescriptorCopies[i].count * sizeof(core::smart_refctd_ptr<const asset::IDescriptor>));
                break;
            }
        }

        assert((descriptorType != asset::EDT_COUNT) && "No binding with binding number pDescriptorCopies[i].binding was found in the set pDescriptorCopies[i].srcSet!");
    }

    // Where do we call the destructors, I think its best to call the destructors in ~IGPUDescriptorSet

    updateDescriptorSets_impl(descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies);

    return true;
}
