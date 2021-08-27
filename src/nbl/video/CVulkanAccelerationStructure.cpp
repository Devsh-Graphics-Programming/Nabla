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

}