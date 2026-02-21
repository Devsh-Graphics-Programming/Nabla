#include "nbl/video/CVulkanSemaphore.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanSemaphore::~CVulkanSemaphore()
{
	const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
	vulkanDevice->getFunctionTable()->vk.vkDestroySemaphore(vulkanDevice->getInternalObject(), m_semaphore, nullptr);
}

uint64_t CVulkanSemaphore::getCounterValue() const
{
	uint64_t retval = 0u;
	const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
	vulkanDevice->getFunctionTable()->vk.vkGetSemaphoreCounterValue(vulkanDevice->getInternalObject(), m_semaphore, &retval);
	return retval;
}

void CVulkanSemaphore::signal(const uint64_t value)
{
	VkSemaphoreSignalInfo info = {VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,nullptr};
	info.semaphore = m_semaphore;
	info.value = value;

	const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
	vulkanDevice->getFunctionTable()->vk.vkSignalSemaphore(vulkanDevice->getInternalObject(), &info);
}

void CVulkanSemaphore::setObjectDebugName(const char* label) const
{
    IBackendObject::setObjectDebugName(label);

	if(!vkSetDebugUtilsObjectNameEXT)
		return;

    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
	VkDebugUtilsObjectNameInfoEXT nameInfo = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, nullptr};
	nameInfo.objectType = VK_OBJECT_TYPE_SEMAPHORE;
	nameInfo.objectHandle = reinterpret_cast<uint64_t>(getInternalObject());
	nameInfo.pObjectName = getObjectDebugName();
	vkSetDebugUtilsObjectNameEXT(vulkanDevice->getInternalObject(), &nameInfo);
}

}