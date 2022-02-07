#include "nbl/video/IPhysicalDevice.h"

using namespace nbl;
using namespace nbl::video;

E_API_TYPE ILogicalDevice::getAPIType() const
{
    return m_physicalDevice->getAPIType();
}

core::smart_refctd_ptr<IGPUDescriptorSetLayout> ILogicalDevice::createGPUDescriptorSetLayout(const IGPUDescriptorSetLayout::SBinding* _begin, const IGPUDescriptorSetLayout::SBinding* _end)
{
    uint32_t dynamicSSBOCount = 0u, dynamicUBOCount = 0u;
    for(auto b = _begin; b != _end; ++b)
    {
        if(b->type == asset::EDT_STORAGE_BUFFER_DYNAMIC)
            dynamicSSBOCount++;
        else if(b->type == asset::EDT_UNIFORM_BUFFER_DYNAMIC)
            dynamicUBOCount++;
        else if(b->type == asset::EDT_COMBINED_IMAGE_SAMPLER && b->samplers)
        {
            auto* samplers = b->samplers;
            for(uint32_t i = 0u; i < b->count; ++i)
                if(!samplers[i]->wasCreatedBy(this))
                    return nullptr;
        }
    }
    const auto& limits = m_physicalDevice->getLimits();
    if(dynamicSSBOCount > limits.maxDynamicOffsetSSBOs || dynamicUBOCount > limits.maxDynamicOffsetUBOs)
        return nullptr;
    return createGPUDescriptorSetLayout_impl(_begin, _end);
}