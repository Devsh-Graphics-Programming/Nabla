#ifndef __NBL_C_VULKAN_DESCRIPTOR_POOL_H_INCLUDED__

#include "nbl/video/IDescriptorPool.h"

#include <volk.h>

namespace nbl::video
{

class ILogicalDevice;

class CVulkanDescriptorPool : public IDescriptorPool
{
public:
    CVulkanDescriptorPool(core::smart_refctd_ptr<ILogicalDevice>&& dev, const IDescriptorPool::E_CREATE_FLAGS flags, uint32_t maxSets, const uint32_t poolSizeCount, const IDescriptorPool::SDescriptorPoolSize* poolSizes, VkDescriptorPool descriptorPool)
        : IDescriptorPool(std::move(dev), flags, maxSets, poolSizeCount, poolSizes), m_descriptorPool(descriptorPool)
    {}

    ~CVulkanDescriptorPool();

    inline VkDescriptorPool getInternalObject() const { return m_descriptorPool; }

    void setObjectDebugName(const char* label) const override;

private:
    bool freeDescriptorSets_impl(const uint32_t descriptorSetCount, IGPUDescriptorSet* const* const descriptorSets) final override;

    VkDescriptorPool m_descriptorPool;
};

}

#define __NBL_C_VULKAN_DESCRIPTOR_POOL_H_INCLUDED__
#endif
