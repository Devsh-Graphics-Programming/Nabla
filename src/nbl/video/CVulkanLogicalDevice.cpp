#include "nbl/video/CVulkanLogicalDevice.h"
#include "nbl/video/CVulkanPhysicalDevice.h"
namespace nbl::video
{
core::smart_refctd_ptr<IGPUAccelerationStructure> CVulkanLogicalDevice::createGPUAccelerationStructure_impl(IGPUAccelerationStructure::SCreationParams&& params) 
{
    auto physicalDevice = static_cast<const CVulkanPhysicalDevice*>(getPhysicalDevice());
    auto features = physicalDevice->getFeatures();
    if(!features.accelerationStructure) // TODO(Erfan): (instead somehow get "Enabled" features) -> Better Conditional
    {
        assert(false && "device acceleration structures is not enabled.");
        return nullptr;
    }

    VkAccelerationStructureKHR vk_as = VK_NULL_HANDLE;
    VkAccelerationStructureCreateInfoKHR vasci = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR, nullptr};
    vasci.createFlags = CVulkanAccelerationStructure::getVkASCreateFlagsFromASCreateFlags(params.flags);
    vasci.type = CVulkanAccelerationStructure::getVkASTypeFromASType(params.type);
    vasci.buffer = static_cast<const CVulkanBuffer*>(params.bufferRange.buffer.get())->getInternalObject();
    vasci.offset = static_cast<VkDeviceSize>(params.bufferRange.offset);
    vasci.size = static_cast<VkDeviceSize>(params.bufferRange.size);
    auto vk_res = vkCreateAccelerationStructureKHR(m_vkdev, &vasci, nullptr, &vk_as);
    if(VK_SUCCESS != vk_res)
        return nullptr;
    return core::make_smart_refctd_ptr<CVulkanAccelerationStructure>(core::smart_refctd_ptr<CVulkanLogicalDevice>(this), std::move(params), vk_as);
}

}