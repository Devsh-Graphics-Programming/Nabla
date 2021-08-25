#ifndef __NBL_C_VULKAN_DESCRIPTOR_POOL_H_INCLUDED__

#include "nbl/video/IDescriptorPool.h"

#include <volk.h>

namespace nbl::video
{

class ILogicalDevice;

class CVulkanDescriptorPool : public IDescriptorPool
{
public:
    CVulkanDescriptorPool(core::smart_refctd_ptr<ILogicalDevice>&& dev, uint32_t maxSets, VkDescriptorPool descriptorPool)
        : IDescriptorPool(std::move(dev), maxSets), m_descriptorPool(descriptorPool)
    {}

    ~CVulkanDescriptorPool();

    inline VkDescriptorPool getInternalObject() const { return m_descriptorPool; }

private:
    VkDescriptorPool m_descriptorPool;
};

}

#define __NBL_C_VULKAN_DESCRIPTOR_POOL_H_INCLUDED__
#endif
