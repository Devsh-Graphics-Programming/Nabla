#ifndef __NBL_C_VULKAN_DESCRIPTOR_POOL_H_INCLUDED__

#include "nbl/video/IDescriptorPool.h"

#include <volk.h>

namespace nbl::video
{

class ILogicalDevice;

class CVulkanDescriptorPool : public IDescriptorPool
{
public:
    CVulkanDescriptorPool(core::smart_refctd_ptr<ILogicalDevice>&& dev, IDescriptorPool::SCreateInfo&& createInfo, VkDescriptorPool descriptorPool)
        : IDescriptorPool(std::move(dev), std::move(createInfo)), m_descriptorPool(descriptorPool)
    {}

    ~CVulkanDescriptorPool();

    inline VkDescriptorPool getInternalObject() const { return m_descriptorPool; }

    void setObjectDebugName(const char* label) const override;

private:
    bool createDescriptorSets_impl(uint32_t count, const IGPUDescriptorSetLayout* const* layouts, SDescriptorOffsets *const offsets, const uint32_t firstSetOffsetInPool, core::smart_refctd_ptr<IGPUDescriptorSet>* output) override;
    bool reset_impl() override;

    VkDescriptorPool m_descriptorPool;
};

}

#define __NBL_C_VULKAN_DESCRIPTOR_POOL_H_INCLUDED__
#endif
