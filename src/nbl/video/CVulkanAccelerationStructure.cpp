#include "nbl/video/CVulkanAccelerationStructure.h"
#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanAccelerationStructure::~CVulkanAccelerationStructure()
{
	const auto originDevice = getOriginDevice();

	if (originDevice->getAPIType() == EAT_VULKAN)
	{
		VkDevice vk_device = static_cast<const CVulkanLogicalDevice*>(originDevice)->getInternalObject();
		vkDestroyAccelerationStructureKHR(vk_device, m_vkAccelerationStructure, nullptr);
	}
}
uint64_t CVulkanAccelerationStructure::getReferenceForDeviceOperations() const 
{
	VkDeviceSize ret = 0;
	const auto originDevice = getOriginDevice();

	if (originDevice->getAPIType() == EAT_VULKAN)
	{
		VkDevice vk_device = static_cast<const CVulkanLogicalDevice*>(originDevice)->getInternalObject();
		VkAccelerationStructureDeviceAddressInfoKHR info = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR, nullptr};
		info.accelerationStructure = m_vkAccelerationStructure;
		ret = vkGetAccelerationStructureDeviceAddressKHR(vk_device, &info);
	}

	return static_cast<uint64_t>(ret);
}
uint64_t CVulkanAccelerationStructure::getReferenceForHostOperations() const
{
	assert(m_vkAccelerationStructure != VK_NULL_HANDLE);
	return reinterpret_cast<uint64_t>(m_vkAccelerationStructure);
}

}