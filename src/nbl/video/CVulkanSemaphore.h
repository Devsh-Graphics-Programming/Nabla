#ifndef __NBL_C_VULKAN_SEMAPHORE_H_INCLUDED__
#define __NBL_C_VULKAN_SEMAPHORE_H_INCLUDED__

#include "nbl/video/IGPUSemaphore.h"

#include <volk.h>

namespace nbl::video
{

class CVKLogicalDevice;

class CVulkanSemaphore final : public IGPUSemaphore
{
public:
    CVulkanSemaphore(CVKLogicalDevice* _vkdev, VkSemaphore semaphore);
    ~CVulkanSemaphore();

    VkSemaphore getInternalObject() const { return m_semaphore; }

private:
    CVKLogicalDevice* m_vkdev;
    VkSemaphore m_semaphore;
};

}

#endif