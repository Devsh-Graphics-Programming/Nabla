#include "CVulkanBufferView.h"

#include "nbl/video/CVulkanCommon.h"
#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanBufferView::~CVulkanBufferView()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroyBufferView(vulkanDevice->getInternalObject(), m_vkBufferView, nullptr);
}

void CVulkanBufferView::setObjectDebugName(const char* label) const
{
    IBackendObject::setObjectDebugName(label);

	// TODO: Check for VK_EXT_debug_marker support

    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
	VkDebugUtilsObjectNameInfoEXT nameInfo = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, nullptr};
	nameInfo.objectType = VK_OBJECT_TYPE_BUFFER_VIEW;
	nameInfo.objectHandle = reinterpret_cast<uint64_t>(getInternalObject());
	nameInfo.pObjectName = getObjectDebugName();
	vkSetDebugUtilsObjectNameEXT(vulkanDevice->getInternalObject(), &nameInfo);
}

}