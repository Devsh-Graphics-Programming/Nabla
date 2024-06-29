#include "nbl/video/IGPUDescriptorSet.h"

#include "nbl/video/IDescriptorPool.h"

namespace nbl::video
{

IGPUDescriptorSet::IGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& layout, core::smart_refctd_ptr<IDescriptorPool>&& pool, IDescriptorPool::SStorageOffsets&& offsets)
    : base_t(std::move(layout)), IBackendObject(std::move(core::smart_refctd_ptr<const ILogicalDevice>(pool->getOriginDevice()))), m_version(0ull), m_pool(std::move(pool)), m_storageOffsets(std::move(offsets))
{
    for (auto i = 0u; i < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++i)
    {
        // There is no descriptor of such type in the set.
        if (m_storageOffsets.data[i] == ~0u)
            continue;

        const auto type = static_cast<asset::IDescriptor::E_TYPE>(i);

        // Default-construct the core::smart_refctd_ptr<IDescriptor>s because even if the user didn't update the descriptor set with ILogicalDevice::updateDescriptorSet we
        // won't have uninitialized memory and destruction wouldn't crash in ~IGPUDescriptorSet.
        auto descriptors = getAllDescriptors(type);
        assert(descriptors);
        std::uninitialized_default_construct_n(descriptors, m_layout->getTotalDescriptorCount(type));
    }

    const auto mutableSamplerCount = m_layout->getTotalMutableCombinedSamplerCount();
    if (mutableSamplerCount > 0)
    {
        auto mutableSamplers = getAllMutableCombinedSamplers();
        assert(mutableSamplers);
        std::uninitialized_default_construct_n(mutableSamplers, mutableSamplerCount);
    }
}

IGPUDescriptorSet::~IGPUDescriptorSet()
{
    if (!isZombie())
        m_pool->deleteSetStorage(m_storageOffsets.getSetOffset());
}

asset::IDescriptor::E_TYPE IGPUDescriptorSet::validateWrite(const IGPUDescriptorSet::SWriteDescriptorSet& write, uint32_t& descriptorRedirectBindingIndex, uint32_t& mutableSamplerRedirectBindingIndex) const
{
    assert(write.dstSet == this);

    const char* debugName = getDebugName();

    // screw it, we'll need to replace the descriptor writing with update templates of descriptor buffer soon anyway
    const auto descriptorType = getBindingType(write.binding, descriptorRedirectBindingIndex);

    auto* descriptors = getDescriptorsIndexed(descriptorType, descriptorRedirectBindingIndex);
    if (!descriptors)
    {
        if (debugName)
            m_pool->m_logger.log("Descriptor set (%s, %p) doesn't allow descriptor of such type at binding %u.", system::ILogger::ELL_ERROR, debugName, this, write.binding);
        else
            m_pool->m_logger.log("Descriptor set (%p) doesn't allow descriptor of such type at binding %u.", system::ILogger::ELL_ERROR, this, write.binding);

        return asset::IDescriptor::E_TYPE::ET_COUNT;
    }

    core::smart_refctd_ptr<video::IGPUSampler>* mutableSamplers = nullptr;
    if (descriptorType == asset::IDescriptor::E_TYPE::ET_SAMPLER or (descriptorType == asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER and write.info->info.image.sampler))
    {
        if (m_layout->getImmutableSamplerRedirect().getCount(IGPUDescriptorSetLayout::CBindingRedirect::binding_number_t{ write.binding }) != 0)
        {
            if (debugName)
                m_pool->m_logger.log("Trying to write samplers at binding %u of Descriptor set (%s, %p), but those are immutable.", system::ILogger::ELL_ERROR, debugName, this, write.binding);
            else
                m_pool->m_logger.log("Trying to write samplers at binding %u of Descriptor set (%p), but those are immutable.", system::ILogger::ELL_ERROR, this, write.binding);
            return asset::IDescriptor::E_TYPE::ET_COUNT;
        }

        for (uint32_t i=0; i<write.count; ++i)
        {
            if (asset::IDescriptor::GetTypeCategory(descriptorType) != write.info[i].desc->getTypeCategory())
            {
                if (debugName)
                    m_pool->m_logger.log("Descriptor set (%s, %p) doesn't allow descriptor of such type category at binding %u.", system::ILogger::ELL_ERROR, debugName, this, write.binding);
                else
                    m_pool->m_logger.log("Descriptor set (%p) doesn't allow descriptor of such type category at binding %u.", system::ILogger::ELL_ERROR, this, write.binding);

                return asset::IDescriptor::E_TYPE::ET_COUNT;
            }
            auto* sampler = descriptorType == asset::IDescriptor::E_TYPE::ET_SAMPLER ? reinterpret_cast<IGPUSampler*>(write.info[i].desc.get()) : write.info[i].info.image.sampler.get();
            if (not sampler) {
                if (debugName)
                    m_pool->m_logger.log("Null sampler provided when trying to write descriptor set (%s, %p). All writes should provide a sampler.", system::ILogger::ELL_ERROR, debugName, write.dstSet);
                else
                    m_pool->m_logger.log("Null sampler provided when trying to write descriptor set (%p). All writes should provide a sampler.", system::ILogger::ELL_ERROR, write.dstSet);
                return asset::IDescriptor::E_TYPE::ET_COUNT;
            }
            else if (not sampler->isCompatibleDevicewise(write.dstSet)) {
                const char* samplerDebugName = sampler->getDebugName();
                if (samplerDebugName && debugName)
                    m_pool->m_logger.log("Sampler (%s, %p) does not exist or is not device-compatible with descriptor set (%s, %p).", system::ILogger::ELL_ERROR, samplerDebugName, sampler, debugName, write.dstSet);
                else
                    m_pool->m_logger.log("Sampler (%p) does not exist or is not device-compatible with descriptor set (%p).", system::ILogger::ELL_ERROR, sampler, write.dstSet);
                return asset::IDescriptor::E_TYPE::ET_COUNT;
            }
        }

        if (descriptorType == asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER) {
            mutableSamplerRedirectBindingIndex = getLayout()->getMutableCombinedSamplerRedirect().findBindingStorageIndex(write.binding).data;
            mutableSamplers = getMutableCombinedSamplersIndexed(mutableSamplerRedirectBindingIndex);
            // Should never reach here, the GetTypeCategory check earlier should ensure that
            if (!mutableSamplers)
            {
                if (debugName)
                    m_pool->m_logger.log("Descriptor set (%s, %p) only allows standalone mutable samplers at binding %u (no combined image samplers).", system::ILogger::ELL_ERROR, debugName, this, write.binding);
                else
                    m_pool->m_logger.log("Descriptor set (%p) only allows standalone mutable samplers at binding %u (no combined image samplers).", system::ILogger::ELL_ERROR, this, write.binding);

                return asset::IDescriptor::E_TYPE::ET_COUNT;
            }
        }
    }

    return descriptorType;
}

void IGPUDescriptorSet::processWrite(const IGPUDescriptorSet::SWriteDescriptorSet& write, const asset::IDescriptor::E_TYPE type, const uint32_t descriptorRedirectBindingIndex, const uint32_t mutableSamplerRedirectBindingIndex)
{
    assert(write.dstSet == this);

    auto* descriptors = getDescriptorsIndexed(type, descriptorRedirectBindingIndex);
    assert(descriptors);

    core::smart_refctd_ptr<video::IGPUSampler>* mutableSamplers = nullptr;
    if (type == asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER
        and write.info->info.image.sampler)
    {
        mutableSamplers = getMutableCombinedSamplersIndexed(mutableSamplerRedirectBindingIndex);
        assert(mutableSamplers);
    }

    for (auto j = 0; j < write.count; ++j)
    {
        descriptors[j] = write.info[j].desc;

        if (mutableSamplers)
            mutableSamplers[j] = write.info[j].info.image.sampler;
    }
    auto& bindingRedirect = m_layout->getDescriptorRedirect(type);
    auto bindingCreateFlags = bindingRedirect.getCreateFlags(redirect_t::storage_range_index_t{ descriptorRedirectBindingIndex });
    if (IGPUDescriptorSetLayout::writeIncrementsVersion(bindingCreateFlags))
        incrementVersion();
}

void IGPUDescriptorSet::dropDescriptors(const IGPUDescriptorSet::SDropDescriptorSet& drop)
{
    assert(drop.dstSet == this);

    uint32_t descriptorRedirectBindingIndex;
    const auto descriptorType = getBindingType(drop.binding, descriptorRedirectBindingIndex);

    auto* dstDescriptors = drop.dstSet->getDescriptorsIndexed(descriptorType, descriptorRedirectBindingIndex);
    if (dstDescriptors)
        for (uint32_t i = 0; i < drop.count; i++)
            dstDescriptors[drop.arrayElement + i] = nullptr;

    if (asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER == descriptorType)
    {
        auto* dstSamplers = drop.dstSet->getMutableCombinedSamplers(drop.binding);
        if (dstSamplers)
            for (uint32_t i = 0; i < drop.count; i++)
                dstSamplers[drop.arrayElement + i] = nullptr;
    }

    // we only increment the version to detect UPDATE-AFTER-BIND and automagically invalidate descriptor sets
    // so, only if we do the path that writes descriptors, do we want to increment version
    auto& bindingRedirect = m_layout->getDescriptorRedirect(descriptorType);
    auto bindingCreateFlags = bindingRedirect.getCreateFlags(redirect_t::storage_range_index_t{ descriptorRedirectBindingIndex });
    if (IGPUDescriptorSetLayout::writeIncrementsVersion(bindingCreateFlags))
        incrementVersion();
}

bool IGPUDescriptorSet::validateCopy(const IGPUDescriptorSet::SCopyDescriptorSet& copy, asset::IDescriptor::E_TYPE& type, uint32_t& srcDescriptorRedirectBindingIndex, uint32_t& dstDescriptorRedirectBindingIndex, uint32_t& srcMutableSamplerRedirectBindingIndex, uint32_t& dstMutableSamplerRedirectBindingIndex) const
{
    assert(copy.dstSet == this);

    const char* srcDebugName = copy.srcSet->getDebugName();
    const char* dstDebugName = copy.dstSet->getDebugName();

    const auto srcLayout = copy.srcSet->getLayout();
    const auto dstLayout = copy.dstSet->getLayout();

    for (uint32_t t = 0; t < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++t)
    {
        type = static_cast<asset::IDescriptor::E_TYPE>(t);

        srcDescriptorRedirectBindingIndex = srcLayout->getDescriptorRedirect(type).findBindingStorageIndex(copy.srcBinding).data;
        dstDescriptorRedirectBindingIndex = dstLayout->getDescriptorRedirect(type).findBindingStorageIndex(copy.dstBinding).data;
        auto* srcDescriptors = copy.srcSet->getDescriptorsIndexed(type, srcDescriptorRedirectBindingIndex);
        auto* dstDescriptors = copy.dstSet->getDescriptorsIndexed(type, dstDescriptorRedirectBindingIndex);

        if (!srcDescriptors)
        {
            assert(!dstDescriptors);
            continue;
        }
        assert(dstDescriptors);

        core::smart_refctd_ptr<IGPUSampler> *srcSamplers = nullptr, *dstSamplers = nullptr;
        if (asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER == type)
        {
            srcMutableSamplerRedirectBindingIndex = srcLayout->getMutableCombinedSamplerRedirect().findBindingStorageIndex(copy.srcBinding).data;
            dstMutableSamplerRedirectBindingIndex = dstLayout->getMutableCombinedSamplerRedirect().findBindingStorageIndex(copy.dstBinding).data;
            srcSamplers = copy.srcSet->getMutableCombinedSamplersIndexed(srcMutableSamplerRedirectBindingIndex);
            dstSamplers = copy.dstSet->getMutableCombinedSamplersIndexed(dstMutableSamplerRedirectBindingIndex);
        }

        if ((!srcDescriptors != !dstDescriptors) || (!srcSamplers != !dstSamplers))
        {
            if (srcDebugName && dstDebugName)
                m_pool->m_logger.log("Incompatible copy from descriptor set (%s, %p) at binding %u to descriptor set (%s, %p) at binding %u.", system::ILogger::ELL_ERROR, srcDebugName, copy.srcSet, copy.srcBinding, dstDebugName, copy.dstSet, copy.dstBinding);
            else
                m_pool->m_logger.log("Incompatible copy from descriptor set (%p) at binding %u to descriptor set (%p) at binding %u.", system::ILogger::ELL_ERROR, copy.srcSet, copy.srcBinding, copy.dstSet, copy.dstBinding);

            return false;
        }
    }

    return true;
}

void IGPUDescriptorSet::processCopy(const IGPUDescriptorSet::SCopyDescriptorSet& copy, const asset::IDescriptor::E_TYPE type, const uint32_t srcDescriptorRedirectBindingIndex, const uint32_t dstDescriptorRedirectBindingIndex, const uint32_t srcMutableSamplerRedirectBindingIndex, const uint32_t dstMutableSamplerRedirectBindingIndex)
{
    assert(copy.dstSet == this);

    auto& bindingRedirect = m_layout->getDescriptorRedirect(type);

    auto* srcDescriptors = copy.srcSet->getDescriptorsIndexed(type, srcDescriptorRedirectBindingIndex);
    auto* dstDescriptors = copy.dstSet->getDescriptorsIndexed(type, dstDescriptorRedirectBindingIndex);

    auto* srcSamplers = type != asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER ? nullptr : copy.srcSet->getMutableCombinedSamplersIndexed(srcMutableSamplerRedirectBindingIndex);
    auto* dstSamplers = type != asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER ? nullptr : copy.dstSet->getMutableCombinedSamplersIndexed(dstMutableSamplerRedirectBindingIndex);

        
    auto bindingCreateFlags = bindingRedirect.getCreateFlags(redirect_t::storage_range_index_t{ dstDescriptorRedirectBindingIndex });
    if (IGPUDescriptorSetLayout::writeIncrementsVersion(bindingCreateFlags))
        incrementVersion();

    std::copy_n(srcDescriptors, copy.count, dstDescriptors);

    if (srcSamplers && dstSamplers)
        std::copy_n(srcSamplers, copy.count, dstSamplers);
}

}