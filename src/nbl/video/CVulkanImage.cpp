#include "CVulkanImage.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanImage::~CVulkanImage()
{
    if (m_vkImage != VK_NULL_HANDLE)
    {
        const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
        auto* vk = vulkanDevice->getFunctionTable();
        vk->vk.vkDestroyImage(vulkanDevice->getInternalObject(), m_vkImage, nullptr);
    }
}

}