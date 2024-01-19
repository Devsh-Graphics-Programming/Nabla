#include "nbl/video/CVulkanImage.h"
#include "nbl/video/CVulkanSwapchain.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanImage::~CVulkanImage()
{
    preDestroyStep();
    // e.g. don't destroy imported handles from the same VkInstance (e.g. if hooking into external Vulkan codebase)
    // truly EXTERNAL_MEMORY imported handles, do need to be destroyed + CloseHandled (separate thing)
    if (!m_cachedCreationParams.skipHandleDestroy)
    {
        const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
        vulkanDevice->getFunctionTable()->vk.vkDestroyImage(vulkanDevice->getInternalObject(),getInternalObject(), nullptr);
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