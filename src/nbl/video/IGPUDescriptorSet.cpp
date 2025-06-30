#include "nbl/video/IGPUDescriptorSet.h"

#include "nbl/video/IDescriptorPool.h"

namespace nbl::video
{

IGPUDescriptorSet::IGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& layout, core::smart_refctd_ptr<IDescriptorPool>&& pool, IDescriptorPool::SStorageOffsets&& offsets)
    : base_t(std::move(layout)), IBackendObject(core::smart_refctd_ptr<const ILogicalDevice>(pool->getOriginDevice())), m_version(0ull), m_pool(std::move(pool)), m_storageOffsets(std::move(offsets))
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

IGPUDescriptorSet::SWriteValidationResult IGPUDescriptorSet::validateWrite(const IGPUDescriptorSet::SWriteDescriptorSet& write) const
{
    assert(write.dstSet == this);

    const char* debugName = getDebugName();
    using redirect_t = IGPUDescriptorSetLayout::CBindingRedirect;
    const redirect_t::binding_number_t bindingNumber(write.binding);

    SWriteValidationResult validationResult;

    // screw it, we'll need to replace the descriptor writing with update templates of descriptor buffer soon anyway
    const auto descriptorType = getBindingType(bindingNumber, validationResult.descriptorRedirectBindingIndex);
    if (asset::IDescriptor::E_TYPE::ET_COUNT == descriptorType)
    {
        if (debugName)
            m_pool->m_logger.log("Descriptor set (%s, %p) has no binding %u.", system::ILogger::ELL_ERROR, debugName, this, write.binding);
        else
            m_pool->m_logger.log("Descriptor set (%p) has no binding %u.", system::ILogger::ELL_ERROR, this, write.binding);

        validationResult.type = asset::IDescriptor::E_TYPE::ET_COUNT;
        return validationResult;
    }

    auto* descriptors = getDescriptors(descriptorType, validationResult.descriptorRedirectBindingIndex);
    // Left this check, but if the above passed then this should never be nullptr I believe
    if (!descriptors)
    {
        if (debugName)
            m_pool->m_logger.log("Descriptor set (%s, %p) doesn't allow descriptor of such type at binding %u.", system::ILogger::ELL_ERROR, debugName, this, write.binding);
        else
            m_pool->m_logger.log("Descriptor set (%p) doesn't allow descriptor of such type at binding %u.", system::ILogger::ELL_ERROR, this, write.binding);

        validationResult.type = asset::IDescriptor::E_TYPE::ET_COUNT;
        return validationResult;
    }

    for (uint32_t i = 0; i < write.count; ++i)
    {
        if (asset::IDescriptor::GetTypeCategory(descriptorType) != write.info[i].desc->getTypeCategory())
        {
            if (debugName)
                m_pool->m_logger.log("Descriptor set (%s, %p) doesn't allow descriptor of such type category at binding %u.", system::ILogger::ELL_ERROR, debugName, this, write.binding);
            else
                m_pool->m_logger.log("Descriptor set (%p) doesn't allow descriptor of such type category at binding %u.", system::ILogger::ELL_ERROR, this, write.binding);

            validationResult.type = asset::IDescriptor::E_TYPE::ET_COUNT;
            return validationResult;
        }
    }

    if (descriptorType == asset::IDescriptor::E_TYPE::ET_SAMPLER or (descriptorType == asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER and write.info->info.combinedImageSampler.sampler))
    {
        if (m_layout->getImmutableSamplerRedirect().getCount(IGPUDescriptorSetLayout::CBindingRedirect::binding_number_t{ write.binding }) != 0)
        {
            if (debugName)
                m_pool->m_logger.log("Trying to write samplers at binding %u of Descriptor set (%s, %p), but those are immutable.", system::ILogger::ELL_ERROR, write.binding, debugName, this);
            else
                m_pool->m_logger.log("Trying to write samplers at binding %u of Descriptor set (%p), but those are immutable.", system::ILogger::ELL_ERROR, write.binding, this);
            validationResult.type = asset::IDescriptor::E_TYPE::ET_COUNT;
            return validationResult;
        }

        for (uint32_t i=0; i<write.count; ++i)
        {
            auto* sampler = descriptorType == asset::IDescriptor::E_TYPE::ET_SAMPLER ? reinterpret_cast<IGPUSampler*>(write.info[i].desc.get()) : write.info[i].info.combinedImageSampler.sampler.get();
            if (not sampler) {
                if (debugName)
                    m_pool->m_logger.log("Null sampler provided when trying to write descriptor set (%s, %p) at binding %u. All writes should provide a sampler.", system::ILogger::ELL_ERROR, debugName, this, write.binding);
                else
                    m_pool->m_logger.log("Null sampler provided when trying to write descriptor set (%p) at binding %u. All writes should provide a sampler.", system::ILogger::ELL_ERROR, this, write.binding);
                validationResult.type = asset::IDescriptor::E_TYPE::ET_COUNT;
                return validationResult;
            }
            if (not sampler->isCompatibleDevicewise(write.dstSet)) {
                const char* samplerDebugName = sampler->getDebugName();
                if (samplerDebugName && debugName)
                    m_pool->m_logger.log("Sampler (%s, %p) does not exist or is not device-compatible with descriptor set (%s, %p).", system::ILogger::ELL_ERROR, samplerDebugName, sampler, debugName, this);
                else
                    m_pool->m_logger.log("Sampler (%p) does not exist or is not device-compatible with descriptor set (%p).", system::ILogger::ELL_ERROR, sampler, this);
                validationResult.type = asset::IDescriptor::E_TYPE::ET_COUNT;
                return validationResult;
            }
        }
        if (descriptorType == asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER) {
            validationResult.mutableSamplerRedirectBindingIndex = getLayout()->getMutableCombinedSamplerRedirect().findBindingStorageIndex(bindingNumber);
            auto mutableSamplers = getMutableCombinedSamplers(validationResult.mutableSamplerRedirectBindingIndex);

            // Should never reach here, the GetTypeCategory check earlier should ensure that
            if (!mutableSamplers)
            {
                if (debugName)
                    m_pool->m_logger.log("Descriptor set (%s, %p) only allows standalone mutable samplers at binding %u (no combined image samplers).", system::ILogger::ELL_ERROR, debugName, this, write.binding);
                else
                    m_pool->m_logger.log("Descriptor set (%p) only allows standalone mutable samplers at binding %u (no combined image samplers).", system::ILogger::ELL_ERROR, this, write.binding);

                validationResult.type = asset::IDescriptor::E_TYPE::ET_COUNT;
                return validationResult;
            }
        }
    }

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#descriptorsets-combinedimagesampler
    if (asset::IDescriptor::GetTypeCategory(descriptorType) == asset::IDescriptor::EC_IMAGE)
    for (uint32_t i = 0; i < write.count; ++i)
    {
        auto layout = write.info[i].info.image.imageLayout;
        if (not (asset::IImage::LAYOUT::GENERAL == layout or asset::IImage::LAYOUT::SHARED_PRESENT == layout or asset::IImage::LAYOUT::READ_ONLY_OPTIMAL == layout))
        {
            if (debugName)
                m_pool->m_logger.log("When writing to descriptor set (%s, %p), an image was provided with a layout that isn't GENERAL, SHARED_PRESENT or READ_ONLY_OPTIMAL", system::ILogger::ELL_ERROR, debugName, this);
            else
                m_pool->m_logger.log("When writing to descriptor set (%p), an image was provided with a layout that isn't GENERAL, SHARED_PRESENT or READ_ONLY_OPTIMAL", system::ILogger::ELL_ERROR, this);
        }
    }

    validationResult.type = descriptorType;
    return validationResult;
}

void IGPUDescriptorSet::processWrite(const IGPUDescriptorSet::SWriteDescriptorSet& write, const SWriteValidationResult& validationResult)
{
    assert(write.dstSet == this);

    auto* descriptors = getDescriptors(validationResult.type, validationResult.descriptorRedirectBindingIndex);
    assert(descriptors);

    core::smart_refctd_ptr<video::IGPUSampler>* mutableSamplers = nullptr;
    if (validationResult.type == asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER
        and write.info->info.combinedImageSampler.sampler)
    {
        mutableSamplers = getMutableCombinedSamplers(validationResult.mutableSamplerRedirectBindingIndex);
        assert(mutableSamplers);
    }

    for (uint32_t j = 0; j < write.count; ++j)
    {
        descriptors[j] = write.info[j].desc;

        if (mutableSamplers)
            mutableSamplers[j] = write.info[j].info.combinedImageSampler.sampler;
    }
    auto& bindingRedirect = m_layout->getDescriptorRedirect(validationResult.type);
    auto bindingCreateFlags = bindingRedirect.getCreateFlags(validationResult.descriptorRedirectBindingIndex);
    if (IGPUDescriptorSetLayout::writeIncrementsVersion(bindingCreateFlags))
        incrementVersion();
}

void IGPUDescriptorSet::dropDescriptors(const IGPUDescriptorSet::SDropDescriptorSet& drop)
{
    assert(drop.dstSet == this);

    using redirect_t = IGPUDescriptorSetLayout::CBindingRedirect;
    const redirect_t::binding_number_t bindingNumber(drop.binding);
    redirect_t::storage_range_index_t descriptorRedirectBindingIndex;
    const auto descriptorType = getBindingType(bindingNumber, descriptorRedirectBindingIndex);

    auto* dstDescriptors = drop.dstSet->getDescriptors(descriptorType, descriptorRedirectBindingIndex);
    if (dstDescriptors)
        for (uint32_t i = 0; i < drop.count; i++)
            dstDescriptors[drop.arrayElement + i] = nullptr;

    if (asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER == descriptorType)
    {
        auto* dstSamplers = drop.dstSet->getMutableCombinedSamplers(bindingNumber);
        if (dstSamplers)
            for (uint32_t i = 0; i < drop.count; i++)
                dstSamplers[drop.arrayElement + i] = nullptr;
    }

    // we only increment the version to detect UPDATE-AFTER-BIND and automagically invalidate descriptor sets
    // so, only if we do the path that writes descriptors, do we want to increment version
    auto& bindingRedirect = m_layout->getDescriptorRedirect(descriptorType);
    auto bindingCreateFlags = bindingRedirect.getCreateFlags(descriptorRedirectBindingIndex);
    if (IGPUDescriptorSetLayout::writeIncrementsVersion(bindingCreateFlags))
        incrementVersion();
}

IGPUDescriptorSet::SCopyValidationResult IGPUDescriptorSet::validateCopy(const IGPUDescriptorSet::SCopyDescriptorSet& copy) const
{
    assert(copy.dstSet == this);

    using redirect_t = IGPUDescriptorSetLayout::CBindingRedirect;
    redirect_t::binding_number_t srcBindingNumber(copy.srcBinding);
    redirect_t::binding_number_t dstBindingNumber(copy.dstBinding);

    const char* srcDebugName = copy.srcSet->getDebugName();
    const char* dstDebugName = copy.dstSet->getDebugName();

    const auto srcLayout = copy.srcSet->getLayout();
    const auto dstLayout = copy.dstSet->getLayout();

    SCopyValidationResult validationResult;

    for (uint32_t t = 0; t < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++t)
    {
        validationResult.type = static_cast<asset::IDescriptor::E_TYPE>(t);

        validationResult.srcDescriptorRedirectBindingIndex = srcLayout->getDescriptorRedirect(validationResult.type).findBindingStorageIndex(srcBindingNumber);
        validationResult.dstDescriptorRedirectBindingIndex = dstLayout->getDescriptorRedirect(validationResult.type).findBindingStorageIndex(dstBindingNumber);
        auto* srcDescriptors = copy.srcSet->getDescriptors(validationResult.type, validationResult.srcDescriptorRedirectBindingIndex);
        auto* dstDescriptors = copy.dstSet->getDescriptors(validationResult.type, validationResult.dstDescriptorRedirectBindingIndex);

        if (!srcDescriptors)
        {
            assert(!dstDescriptors);
            continue;
        }
        assert(dstDescriptors);

        core::smart_refctd_ptr<IGPUSampler> *srcSamplers = nullptr, *dstSamplers = nullptr;
        if (asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER == validationResult.type)
        {
            validationResult.srcMutableSamplerRedirectBindingIndex = srcLayout->getMutableCombinedSamplerRedirect().findBindingStorageIndex(srcBindingNumber);
            validationResult.dstMutableSamplerRedirectBindingIndex = dstLayout->getMutableCombinedSamplerRedirect().findBindingStorageIndex(dstBindingNumber);
            srcSamplers = copy.srcSet->getMutableCombinedSamplers(validationResult.srcMutableSamplerRedirectBindingIndex);
            dstSamplers = copy.dstSet->getMutableCombinedSamplers(validationResult.dstMutableSamplerRedirectBindingIndex);
        }

        if ((!srcDescriptors != !dstDescriptors) || (!srcSamplers != !dstSamplers))
        {
            if (srcDebugName && dstDebugName)
                m_pool->m_logger.log("Incompatible copy from descriptor set (%s, %p) at binding %u to descriptor set (%s, %p) at binding %u.", system::ILogger::ELL_ERROR, srcDebugName, copy.srcSet, copy.srcBinding, dstDebugName, copy.dstSet, copy.dstBinding);
            else
                m_pool->m_logger.log("Incompatible copy from descriptor set (%p) at binding %u to descriptor set (%p) at binding %u.", system::ILogger::ELL_ERROR, copy.srcSet, copy.srcBinding, copy.dstSet, copy.dstBinding);

            validationResult.type = asset::IDescriptor::E_TYPE::ET_COUNT;
            return validationResult;
        }
    }

    return validationResult;
}

void IGPUDescriptorSet::processCopy(const IGPUDescriptorSet::SCopyDescriptorSet& copy, const SCopyValidationResult& validationResult)
{
    assert(copy.dstSet == this);

    auto& bindingRedirect = m_layout->getDescriptorRedirect(validationResult.type);

    auto* srcDescriptors = copy.srcSet->getDescriptors(validationResult.type, validationResult.srcDescriptorRedirectBindingIndex);
    auto* dstDescriptors = copy.dstSet->getDescriptors(validationResult.type, validationResult.dstDescriptorRedirectBindingIndex);

    auto* srcSamplers = validationResult.type != asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER ? nullptr : copy.srcSet->getMutableCombinedSamplers(validationResult.srcMutableSamplerRedirectBindingIndex);
    auto* dstSamplers = validationResult.type != asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER ? nullptr : copy.dstSet->getMutableCombinedSamplers(validationResult.dstMutableSamplerRedirectBindingIndex);
       
    auto bindingCreateFlags = bindingRedirect.getCreateFlags(validationResult.dstDescriptorRedirectBindingIndex);
    if (IGPUDescriptorSetLayout::writeIncrementsVersion(bindingCreateFlags))
        incrementVersion();

    std::copy_n(srcDescriptors, copy.count, dstDescriptors);

    if (srcSamplers && dstSamplers)
        std::copy_n(srcSamplers, copy.count, dstSamplers);
}

}