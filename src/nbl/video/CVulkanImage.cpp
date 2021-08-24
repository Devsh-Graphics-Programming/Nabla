#include "CVulkanImage.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanImage::~CVulkanImage()
{
    const auto originDevice = getOriginDevice();

    if (originDevice->getAPIType() == EAT_VULKAN)
    {
        // auto* vk = m_vkdevice->getFunctionTable();
        // vk->vk.vkDestroyImage(vkdev, m_vkimg, nullptr);
        if (m_vkImage != VK_NULL_HANDLE)
        {
            VkDevice vk_device = static_cast<const CVulkanLogicalDevice*>(originDevice)->getInternalObject();
            vkDestroyImage(vk_device, m_vkImage, nullptr);
        }
    }
}

}