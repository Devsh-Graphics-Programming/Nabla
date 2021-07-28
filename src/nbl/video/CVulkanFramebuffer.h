#ifndef __NBL_C_VULKAN_FRAMEBUFFER_H_INCLUDED__
#define __NBL_C_VULKAN_FRAMEBUFFER_H_INCLUDED__

#include "nbl/video/IGPUFramebuffer.h"

#include <volk.h>

namespace nbl::video
{

class CVKLogicalDevice;

class CVulkanFramebuffer final : public IGPUFramebuffer
{
public:
    CVulkanFramebuffer(CVKLogicalDevice* vkdev, SCreationParams&& params);
    ~CVulkanFramebuffer();

private:
    CVKLogicalDevice* m_vkdevice;
    VkFramebuffer m_vkfbo;
};

}

#endif