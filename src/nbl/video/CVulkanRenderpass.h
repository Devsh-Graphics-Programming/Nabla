#ifndef __NBL_C_VULKAN_RENDERPASS_H_INCLUDED__
#define __NBL_C_VULKAN_RENDERPASS_H_INCLUDED__

#include "nbl/video/IGPURenderpass.h"

#include <volk.h>

namespace nbl {
namespace video
{

class CVKLogicalDevice;

class CVulkanRenderpass final : public IGPURenderpass
{
    explicit CVulkanRenderpass(const SCreationParams& params);

    ~CVulkanRenderpass();

private:
    CVKLogicalDevice* m_vkdev;
    VkRenderPass m_renderpass;
};

}
}

#endif
