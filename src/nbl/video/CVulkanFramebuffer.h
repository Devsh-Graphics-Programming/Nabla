#ifndef __NBL_C_VULKAN_FRAMEBUFFER_H_INCLUDED__
#define __NBL_C_VULKAN_FRAMEBUFFER_H_INCLUDED__

#include "nbl/video/IFramebuffer.h"

#include <volk.h>

namespace nbl {
namespace video
{

class CVKLogicalDevice;

class CVulkanFramebuffer final : public IFramebuffer
{
public:
    CVulkanFramebuffer(CVKLogicalDevice* vkdev, SCreationParams&& params);
    ~CVulkanFramebuffer();

private:
    CVKLogicalDevice* m_vkdevice;
    VkFramebuffer m_vkfbo;
};

}}

#endif