#ifndef __NBL_C_VULKAN_FENCE_H_INCLUDED__
#define __NBL_C_VULKAN_FENCE_H_INCLUDED__

#include "nbl/video/IGPUFence.h"

#include <volk.h>

namespace nbl::video
{

class ILogicalDevice;

class CVulkanFence final : public IGPUFence
{
public:
    CVulkanFence(core::smart_refctd_ptr<ILogicalDevice>&& _vkdev, E_CREATE_FLAGS _flags,
        VkFence fence) : IGPUFence(std::move(_vkdev), _flags), m_fence(fence)
    {}

    ~CVulkanFence();

    VkFence getInternalObject() const { return m_fence; }

private:
    VkFence m_fence;
};

}

#endif
