#include "nbl/video/CVulkanImage.h"
#include "nbl/video/CVulkanSwapchain.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanImage::~CVulkanImage()
{
    if (m_optionalBackingSwapchain)
    {
        // This is a swapchain image, we will drop the ref to the swapchain and set the corresponding imageExists to false
        auto vkswapchain = core::smart_refctd_ptr_static_cast<CVulkanSwapchain>(m_optionalBackingSwapchain);
        vkswapchain->m_imageExists->begin()[m_optionalIndexWithinSwapchain].store(false, std::memory_order_release);
    }
    // Note: we don't destroy a swapchain image as that's invalid
    else
    {
        if (m_vkImage != VK_NULL_HANDLE)
        {
            const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
            auto* vk = vulkanDevice->getFunctionTable();
            vk->vk.vkDestroyImage(vulkanDevice->getInternalObject(), m_vkImage, nullptr);
        }
    }
}

void CVulkanImage::setObjectDebugName(const char* label) const
{
    IBackendObject::setObjectDebugName(label);

	if(vkSetDebugUtilsObjectNameEXT == 0) return;

    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
	VkDebugUtilsObjectNameInfoEXT nameInfo = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, nullptr};
	nameInfo.objectType = VK_OBJECT_TYPE_IMAGE;
	nameInfo.objectHandle = reinterpret_cast<uint64_t>(getInternalObject());
	nameInfo.pObjectName = getObjectDebugName();
	vkSetDebugUtilsObjectNameEXT(vulkanDevice->getInternalObject(), &nameInfo);
}

}