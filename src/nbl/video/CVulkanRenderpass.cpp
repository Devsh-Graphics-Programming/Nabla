#include "nbl/video/CVulkanRenderpass.h"

#include "nbl/video/CVKLogicalDevice.h"

namespace nbl::video
{

CVulkanRenderpass::~CVulkanRenderpass()
{
    const auto originDevice = getOriginDevice();

    if (originDevice->getAPIType() == EAT_VULKAN)
    {
        // auto* vk = m_vkdev->getFunctionTable();
        VkDevice vk_device = static_cast<const CVKLogicalDevice*>(originDevice)->getInternalObject();
        // vk->vk.vkDestroyRenderPass(vkdev, m_renderpass, nullptr);
        vkDestroyRenderPass(vk_device, m_renderpass, nullptr);
    }
}

}