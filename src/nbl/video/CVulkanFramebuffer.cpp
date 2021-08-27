#include "nbl/video/CVulkanFramebuffer.h"

#include "nbl/video/CVulkanLogicalDevice.h"
#include "nbl/video/CVulkanRenderpass.h"
#include "nbl/video/CVulkanImageView.h"

namespace nbl::video
{

CVulkanFramebuffer::~CVulkanFramebuffer()
{
    const auto originDevice = getOriginDevice();

    if (originDevice->getAPIType() == EAT_VULKAN)
    {
        // auto* vk = m_vkdev->getFunctionTable();
        VkDevice vk_device = static_cast<const CVulkanLogicalDevice*>(originDevice)->getInternalObject();
        // vk->vk.vkDestroyFramebuffer(vkdev, m_vkfbo, nullptr);
        vkDestroyFramebuffer(vk_device, m_vkfbo, nullptr);
    }
}

}