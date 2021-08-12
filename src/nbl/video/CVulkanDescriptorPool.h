#ifndef __NBL_C_VULKAN_DESCRIPTOR_POOL_H_INCLUDED__

#include "nbl/video/IDescriptorPool.h"

#include <volk.h>

namespace nbl::video
{

class ILogicalDevice;

class CVulkanDescriptorPool : public IDescriptorPool
{
public:
    CVulkanDescriptorPool(ILogicalDevice* dev, VkDescriptorPool descriptorPool)
        : IDescriptorPool(dev), m_descriptorPool(descriptorPool)
    {}

    ~CVulkanDescriptorPool();

    inline VkDescriptorPool getInternalObject() const { return m_descriptorPool; }

private:
    VkDescriptorPool m_descriptorPool;
};

}

#define __NBL_C_VULKAN_DESCRIPTOR_POOL_H_INCLUDED__
#endif
