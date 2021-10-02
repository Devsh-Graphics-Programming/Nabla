#include "nbl/video/CVulkanRenderpass.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanRenderpass::~CVulkanRenderpass()
{
    // auto* vk = m_vkdev->getFunctionTable();
    VkDevice vk_device = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getInternalObject();
    // vk->vk.vkDestroyRenderPass(vkdev, m_renderpass, nullptr);
    vkDestroyRenderPass(vk_device, m_renderpass, nullptr);
}

}