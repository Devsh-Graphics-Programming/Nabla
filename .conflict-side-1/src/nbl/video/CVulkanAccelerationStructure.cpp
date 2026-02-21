#include "nbl/video/CVulkanAccelerationStructure.h"
#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

template<class GPUAccelerationStructure>
CVulkanAccelerationStructure<GPUAccelerationStructure>::CVulkanAccelerationStructure(core::smart_refctd_ptr<const CVulkanLogicalDevice>&& logicalDevice, GPUAccelerationStructure::SCreationParams&& params, const VkAccelerationStructureKHR accelerationStructure)
	: GPUAccelerationStructure(std::move(logicalDevice),std::move(params)), m_vkAccelerationStructure(accelerationStructure)
{
	auto* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(GPUAccelerationStructure::getOriginDevice());
	VkAccelerationStructureDeviceAddressInfoKHR info = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,nullptr};
	info.accelerationStructure = m_vkAccelerationStructure;
	m_deviceAddress = vulkanDevice->getFunctionTable()->vk.vkGetAccelerationStructureDeviceAddressKHR(vulkanDevice->getInternalObject(),&info);
}

template<class GPUAccelerationStructure>
CVulkanAccelerationStructure<GPUAccelerationStructure>::~CVulkanAccelerationStructure()
{
	auto* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(GPUAccelerationStructure::getOriginDevice());
	vulkanDevice->getFunctionTable()->vk.vkDestroyAccelerationStructureKHR(vulkanDevice->getInternalObject(),m_vkAccelerationStructure,nullptr);
}

template<class GPUAccelerationStructure>
bool CVulkanAccelerationStructure<GPUAccelerationStructure>::wasCopySuccessful(const IDeferredOperation* const deferredOp)
{
	auto* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(GPUAccelerationStructure::getOriginDevice());
	return vulkanDevice->getFunctionTable()->vk.vkGetDeferredOperationResultKHR(vulkanDevice->getInternalObject(),static_cast<const CVulkanDeferredOperation*>(deferredOp)->getInternalObject())==VK_SUCCESS;
}

template<class GPUAccelerationStructure>
bool CVulkanAccelerationStructure<GPUAccelerationStructure>::wasBuildSuccessful(const IDeferredOperation* const deferredOp)
{
	auto* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(GPUAccelerationStructure::getOriginDevice());
	return vulkanDevice->getFunctionTable()->vk.vkGetDeferredOperationResultKHR(vulkanDevice->getInternalObject(),static_cast<const CVulkanDeferredOperation*>(deferredOp)->getInternalObject())==VK_SUCCESS;
}

template class CVulkanAccelerationStructure<IGPUBottomLevelAccelerationStructure>;
template class CVulkanAccelerationStructure<IGPUTopLevelAccelerationStructure>;

}