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

    inline void* getNativeHandle() override {return &m_fence;}
    VkFence getInternalObject() const {return m_fence;}

    void setObjectDebugName(const char* label) const override;

private:
    VkFence m_fence;
};

}

#endif
