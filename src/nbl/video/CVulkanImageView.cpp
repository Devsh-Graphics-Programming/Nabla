#include "nbl/video/CVulkanImageView.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanImageView::~CVulkanImageView()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroyImageView(vulkanDevice->getInternalObject(), m_vkImageView, nullptr);
}

}