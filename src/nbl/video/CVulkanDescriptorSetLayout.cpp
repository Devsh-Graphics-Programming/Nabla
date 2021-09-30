#include "CVulkanDescriptorSetLayout.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{
CVulkanDescriptorSetLayout::~CVulkanDescriptorSetLayout()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroyDescriptorSetLayout(vulkanDevice->getInternalObject(), m_dsLayout, nullptr);
}
void CVulkanDescriptorSetLayout::setObjectDebugName(const char* label) const
{
    IBackendObject::setObjectDebugName(label);

	// TODO: Check for VK_EXT_debug_marker support

    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
	VkDebugUtilsObjectNameInfoEXT nameInfo = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, nullptr};
	nameInfo.objectType = VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT;
	nameInfo.objectHandle = reinterpret_cast<uint64_t>(getInternalObject());
	nameInfo.pObjectName = getObjectDebugName();
	vkSetDebugUtilsObjectNameEXT(vulkanDevice->getInternalObject(), &nameInfo);
}
}