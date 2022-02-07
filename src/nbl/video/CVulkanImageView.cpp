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

void CVulkanImageView::setObjectDebugName(const char* label) const
{
    IBackendObject::setObjectDebugName(label);

    if(vkSetDebugUtilsObjectNameEXT == 0)
        return;

    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    VkDebugUtilsObjectNameInfoEXT nameInfo = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, nullptr};
    nameInfo.objectType = VK_OBJECT_TYPE_IMAGE_VIEW;
    nameInfo.objectHandle = reinterpret_cast<uint64_t>(getInternalObject());
    nameInfo.pObjectName = getObjectDebugName();
    vkSetDebugUtilsObjectNameEXT(vulkanDevice->getInternalObject(), &nameInfo);
}

}