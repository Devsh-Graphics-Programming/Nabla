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

template<>
static VkDeviceOrHostAddressKHR CVulkanAccelerationStructure::getVkDeviceOrHostAddr(VkDevice vk_device, const DeviceAddressType & addr) {
	VkDeviceOrHostAddressKHR ret = {};
	if(addr.buffer.get() != nullptr)
	{
		VkBuffer vk_buffer = static_cast<const CVulkanBuffer*>(addr.buffer.get())->getInternalObject();
		assert(vk_buffer != VK_NULL_HANDLE);
		VkBufferDeviceAddressInfo info = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr};
		info.buffer = vk_buffer;
		ret.deviceAddress = vkGetBufferDeviceAddressKHR(vk_device, &info) + addr.offset;
	}
	return ret;
}
template<>
static VkDeviceOrHostAddressKHR CVulkanAccelerationStructure::getVkDeviceOrHostAddr(VkDevice vk_device, const HostAddressType & addr) {
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
static VkDeviceOrHostAddressConstKHR CVulkanAccelerationStructure::getVkDeviceOrHostConstAddr(VkDevice vk_device, const DeviceAddressType & addr) {
	VkDeviceOrHostAddressConstKHR ret = {};
	if(addr.buffer.get() != nullptr)
	{
		VkBuffer vk_buffer = static_cast<const CVulkanBuffer*>(addr.buffer.get())->getInternalObject();
		assert(vk_buffer != VK_NULL_HANDLE);
		VkBufferDeviceAddressInfo info = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr};
		info.buffer = vk_buffer;
		ret.deviceAddress = vkGetBufferDeviceAddressKHR(vk_device, &info) + addr.offset;
	}
	return ret;
}
template<>
static VkDeviceOrHostAddressConstKHR CVulkanAccelerationStructure::getVkDeviceOrHostConstAddr(VkDevice vk_device, const HostAddressType & addr) {
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