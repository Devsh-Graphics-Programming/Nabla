#include "nbl/video/CVulkanImageView.h"

#include "nbl/video/CVKLogicalDevice.h"

namespace nbl::video
{

CVulkanImageView::~CVulkanImageView()
{
    const auto originDevice = getOriginDevice();

    if (originDevice->getAPIType() == EAT_VULKAN)
    {
        // auto* vk = m_vkdev->getFunctionTable();
        VkDevice vk_device = static_cast<const CVKLogicalDevice*>(originDevice)->getInternalObject();
        // vk->vk.vkDestroyImageView(vkdev, m_vkimgview, nullptr);
        vkDestroyImageView(vk_device, m_vkImageView, nullptr);
    }
}

}