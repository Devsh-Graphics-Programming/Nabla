#include "nbl/video/CVulkanCommandPool.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanCommandPool::~CVulkanCommandPool()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroyCommandPool(vulkanDevice->getInternalObject(), m_vkCommandPool, nullptr);
}

void CVulkanCommandPool::setObjectDebugName(const char* label) const
{
    IBackendObject::setObjectDebugName(label);

	if(vkSetDebugUtilsObjectNameEXT == 0) return;

    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
	VkDebugUtilsObjectNameInfoEXT nameInfo = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, nullptr};
	nameInfo.objectType = VK_OBJECT_TYPE_COMMAND_POOL;
	nameInfo.objectHandle = reinterpret_cast<uint64_t>(getInternalObject());
	nameInfo.pObjectName = getObjectDebugName();
	vkSetDebugUtilsObjectNameEXT(vulkanDevice->getInternalObject(), &nameInfo);
}

bool CVulkanCommandPool::reset_impl()
{
    const auto* vk_device = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    const auto vk = vk_device->getFunctionTable();
    const VkResult result = vk->vk.vkResetCommandPool(*((VkDevice*)vk_device->getNativeHandle()), m_vkCommandPool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
    return result == VK_SUCCESS;
}

}