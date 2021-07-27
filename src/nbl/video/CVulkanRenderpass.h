#ifndef __NBL_C_VULKAN_RENDERPASS_H_INCLUDED__
#define __NBL_C_VULKAN_RENDERPASS_H_INCLUDED__

#include "nbl/video/IGPURenderpass.h"

#include <volk.h>

namespace nbl::video
{
class CVKLogicalDevice;

class CVulkanRenderpass final : public IGPURenderpass
{
public:
    explicit CVulkanRenderpass(CVKLogicalDevice* logicalDevice, const SCreationParams& params);

    ~CVulkanRenderpass();

    VkRenderPass getInternalObject() const { return m_renderpass; }

private:
    CVKLogicalDevice* m_vkdev;
    VkRenderPass m_renderpass;
};

}

#endif
