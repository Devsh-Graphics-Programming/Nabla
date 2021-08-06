#ifndef __NBL_C_VULKAN_FENCE_H_INCLUDED__
#define __NBL_C_VULKAN_FENCE_H_INCLUDED__

#include "nbl/video/IGPUFence.h"

#include <volk.h>

namespace nbl::video
{

class CVKLogicalDevice;

class CVulkanFence final : public IGPUFence
{
public:
    CVulkanFence(CVKLogicalDevice* _vkdev, E_CREATE_FLAGS _flags, VkFence fence);
    ~CVulkanFence();

    VkFence getInternalObject() const { return m_fence; }

private:
    CVKLogicalDevice* m_vkdev;
    VkFence m_fence;
};

}

#endif
