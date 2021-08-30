#ifndef __NBL_C_VULKAN_SEMAPHORE_H_INCLUDED__
#define __NBL_C_VULKAN_SEMAPHORE_H_INCLUDED__

#include "nbl/video/IGPUSemaphore.h"

#include <volk.h>

namespace nbl::video
{

class ILogicalDevice;

class CVulkanSemaphore final : public IGPUSemaphore
{
public:
    CVulkanSemaphore(core::smart_refctd_ptr<ILogicalDevice>&& _vkdev,
        VkSemaphore semaphore) : IGPUSemaphore(std::move(_vkdev)), m_semaphore(semaphore)
    {}

    ~CVulkanSemaphore();

    VkSemaphore getInternalObject() const { return m_semaphore; }

private:
    VkSemaphore m_semaphore;
};

}

#endif