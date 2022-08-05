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

void CVulkanCommandPool::reset()
{
	// Free everything in the memory pool from all command buffers
	// TODO figure out if this is the best way to do it
	mempool.reset();

	const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
	auto* vk = vulkanDevice->getFunctionTable();
	vk->vk.vkResetCommandPool(vulkanDevice->getInternalObject(), m_vkCommandPool, 0);
	IGPUCommandPool::reset();
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

}