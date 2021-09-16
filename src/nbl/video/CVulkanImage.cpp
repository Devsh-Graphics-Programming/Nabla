#include "CVulkanImage.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanImage::~CVulkanImage()
{
    if (m_vkImage != VK_NULL_HANDLE)
    {
        // auto* vk = m_vkdevice->getFunctionTable();
        // vk->vk.vkDestroyImage(vkdev, m_vkimg, nullptr);
        VkDevice vk_device = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getInternalObject();
        vkDestroyImage(vk_device, m_vkImage, nullptr);
    }
}

}