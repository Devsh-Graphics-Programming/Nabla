#include "nbl/video/IPhysicalDevice.h"

using namespace nbl;
using namespace nbl::video;


E_API_TYPE ILogicalDevice::getAPIType() const { return m_physicalDevice->getAPIType(); }

void ILogicalDevice::addJITIncludeLoader()
{
    if(auto cc = (m_physicalDevice && m_compilerSet) ? m_compilerSet->getShaderCompiler(asset::IShader::E_CONTENT_TYPE::ECT_HLSL) : nullptr)
    {
        if(auto finder = cc->getDefaultIncludeFinder())
        {
            finder->addSearchPath("nbl/builtin/hlsl/jit", core::make_smart_refctd_ptr<CJITIncludeLoader>(m_physicalDevice->getLimits(), m_physicalDevice->getFeatures()));
        }
    }
}

core::smart_refctd_ptr<IGPUDescriptorSetLayout> ILogicalDevice::createDescriptorSetLayout(const IGPUDescriptorSetLayout::SBinding* _begin, const IGPUDescriptorSetLayout::SBinding* _end)
{
    uint32_t dynamicSSBOCount=0u,dynamicUBOCount=0u;
    for (auto b=_begin; b!=_end; ++b)
    {
        if (b->type == asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER_DYNAMIC)
            dynamicSSBOCount++;
        else if (b->type == asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER_DYNAMIC)
            dynamicUBOCount++;
        else if (b->type == asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER && b->samplers)
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
        const auto& write = pDescriptorWrites[i];
        auto* ds = static_cast<IGPUDescriptorSet*>(write.dstSet);

        assert(ds->getLayout()->isCompatibleDevicewise(ds));

        if (!ds->validateWrite(write))
            return false;
    }

    for (auto i = 0; i < descriptorCopyCount; ++i)
    {
        const auto& copy = pDescriptorCopies[i];
        const auto* srcDS = static_cast<const IGPUDescriptorSet*>(copy.srcSet);
        const auto* dstDS = static_cast<IGPUDescriptorSet*>(copy.dstSet);

        if (!dstDS->isCompatibleDevicewise(srcDS))
            return false;

        if (!dstDS->validateCopy(copy))
            return false;
    }

    for (auto i = 0; i < descriptorWriteCount; ++i)
    {
        auto& write = pDescriptorWrites[i];
        auto* ds = static_cast<IGPUDescriptorSet*>(write.dstSet);
        ds->processWrite(write);
    }

    for (auto i = 0; i < descriptorCopyCount; ++i)
    {
        const auto& copy = pDescriptorCopies[i];
        auto* dstDS = static_cast<IGPUDescriptorSet*>(pDescriptorCopies[i].dstSet);
        dstDS->processCopy(copy);
    }

    updateDescriptorSets_impl(descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies);

    return true;
}
