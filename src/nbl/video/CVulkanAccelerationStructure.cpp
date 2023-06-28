#include "nbl/video/CVulkanAccelerationStructure.h"
#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanAccelerationStructure::CVulkanAccelerationStructure(core::smart_refctd_ptr<const ILogicalDevice>&& logicalDevice, SCreationParams&& _params, VkAccelerationStructureKHR accelerationStructure)
	: IGPUAccelerationStructure(std::move(logicalDevice),std::move(_params)), m_vkAccelerationStructure(accelerationStructure)
{
	VkAccelerationStructureDeviceAddressInfoKHR info = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,nullptr};
	info.accelerationStructure = m_vkAccelerationStructure;
	const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
	m_deviceAddress = vulkanDevice->getFunctionTable()->vk.vkGetAccelerationStructureDeviceAddressKHR(vulkanDevice->getInternalObject(),&info);
}

CVulkanAccelerationStructure::~CVulkanAccelerationStructure()
{
	const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
	vulkanDevice->getFunctionTable()->vk.vkDestroyAccelerationStructureKHR(vulkanDevice->getInternalObject(),m_vkAccelerationStructure,nullptr);
}

template<>
VkDeviceOrHostAddressKHR CVulkanAccelerationStructure::getVkDeviceOrHostAddress(VkDevice vk_device, const CVulkanDeviceFunctionTable* vk_devf, const DeviceAddressType& addr) {
	VkDeviceOrHostAddressKHR ret = {};
	if(addr.buffer.get() != nullptr)
	{
		VkBuffer vk_buffer = static_cast<const CVulkanBuffer*>(addr.buffer.get())->getInternalObject();
		assert(vk_buffer != VK_NULL_HANDLE);
		VkBufferDeviceAddressInfo info = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr};
		info.buffer = vk_buffer;
		ret.deviceAddress = vk_devf->vk.vkGetBufferDeviceAddressKHR(vk_device, &info) + addr.offset;
	}
	return ret;
}
template<>
VkDeviceOrHostAddressKHR CVulkanAccelerationStructure::getVkDeviceOrHostAddress(VkDevice vk_device, const CVulkanDeviceFunctionTable* vk_devf, const HostAddressType& addr) {
	VkDeviceOrHostAddressKHR ret = {};
	if(addr.buffer.get() != nullptr) 
	{
		if(addr.buffer->getPointer() != nullptr) 
		{
			uint8_t* hostData = reinterpret_cast<uint8_t*>(addr.buffer->getPointer()) + addr.offset;
			ret.hostAddress = hostData;
		}
	}
	return ret;
}
template<>
VkDeviceOrHostAddressConstKHR CVulkanAccelerationStructure::getVkDeviceOrHostConstAddress(VkDevice vk_device, const CVulkanDeviceFunctionTable* vk_devf, const DeviceAddressType& addr) {
	VkDeviceOrHostAddressConstKHR ret = {};
	if(addr.buffer.get() != nullptr)
	{
		VkBuffer vk_buffer = static_cast<const CVulkanBuffer*>(addr.buffer.get())->getInternalObject();
		assert(vk_buffer != VK_NULL_HANDLE);
		VkBufferDeviceAddressInfo info = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr};
		info.buffer = vk_buffer;
		ret.deviceAddress = vk_devf->vk.vkGetBufferDeviceAddressKHR(vk_device, &info) + addr.offset;
	}
	return ret;
}
template<>
VkDeviceOrHostAddressConstKHR CVulkanAccelerationStructure::getVkDeviceOrHostConstAddress(VkDevice vk_device, const CVulkanDeviceFunctionTable* vk_devf, const HostAddressType& addr) {
	VkDeviceOrHostAddressConstKHR ret = {};
	if(addr.buffer.get() != nullptr) 
	{
		if(addr.buffer->getPointer() != nullptr) 
		{
			uint8_t* hostData = reinterpret_cast<uint8_t*>(addr.buffer->getPointer()) + addr.offset;
			ret.hostAddress = hostData;
		}
	}
	return ret;
}

}