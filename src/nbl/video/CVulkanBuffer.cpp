#include "CVulkanBuffer.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanBuffer::~CVulkanBuffer()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroyBuffer(vulkanDevice->getInternalObject(), m_vkBuffer, nullptr);
}

void CVulkanBuffer::setObjectDebugName(const char* label) const
{
    IBackendObject::setObjectDebugName(label);

	// TODO: Check for VK_EXT_debug_marker support

    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
	VkDebugUtilsObjectNameInfoEXT nameInfo = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, nullptr};
	nameInfo.objectType = VK_OBJECT_TYPE_BUFFER;
	nameInfo.objectHandle = reinterpret_cast<uint64_t>(getInternalObject());
	nameInfo.pObjectName = getObjectDebugName();
	vkSetDebugUtilsObjectNameEXT(vulkanDevice->getInternalObject(), &nameInfo);
}

}