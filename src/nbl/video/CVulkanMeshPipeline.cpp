#include "nbl/video/CVulkanMeshPipeline.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

	CVulkanMeshPipeline::~CVulkanMeshPipeline()
	{
		const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
		auto* vk = vulkanDevice->getFunctionTable();
		vk->vk.vkDestroyPipeline(vulkanDevice->getInternalObject(), m_vkPipeline, nullptr);
	}
	void CVulkanMeshPipeline::setObjectDebugName(const char* label) const
	{
		IBackendObject::setObjectDebugName(label);

		if (vkSetDebugUtilsObjectNameEXT == 0) return;

		const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
		VkDebugUtilsObjectNameInfoEXT nameInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, nullptr };
		nameInfo.objectType = VK_OBJECT_TYPE_PIPELINE;
		nameInfo.objectHandle = reinterpret_cast<uint64_t>(getInternalObject());
		nameInfo.pObjectName = getObjectDebugName();
		vkSetDebugUtilsObjectNameEXT(vulkanDevice->getInternalObject(), &nameInfo);
	}
}