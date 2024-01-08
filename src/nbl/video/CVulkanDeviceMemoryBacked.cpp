#define _NBL_VIDEO_C_VULKAN_DEVICE_MEMORY_BACKED_CPP_
#include "nbl/video/CVulkanDeviceMemoryBacked.h"
#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

template<class Interface>
IDeviceMemoryBacked::SDeviceMemoryRequirements CVulkanDeviceMemoryBacked<Interface>::obtainRequirements(const CVulkanLogicalDevice* device, const VkResource_t vkHandle)
{    
    const std::conditional_t<IsImage,VkImageMemoryRequirementsInfo2,VkBufferMemoryRequirementsInfo2> vk_memoryRequirementsInfo = {
        IsImage ? VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2:VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,nullptr,vkHandle
    };

    VkMemoryDedicatedRequirements vk_dedicatedMemoryRequirements = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,nullptr };
    VkMemoryRequirements2 vk_memoryRequirements = { VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,&vk_dedicatedMemoryRequirements };
    const auto& vk = device->getFunctionTable()->vk;
    if constexpr(IsImage)
        vk.vkGetImageMemoryRequirements2(device->getInternalObject(),&vk_memoryRequirementsInfo,&vk_memoryRequirements);
    else
        vk.vkGetBufferMemoryRequirements2(device->getInternalObject(),&vk_memoryRequirementsInfo,&vk_memoryRequirements);

    IDeviceMemoryBacked::SDeviceMemoryRequirements memoryReqs = {};
    memoryReqs.size = vk_memoryRequirements.memoryRequirements.size;
    memoryReqs.memoryTypeBits = vk_memoryRequirements.memoryRequirements.memoryTypeBits;
    memoryReqs.alignmentLog2 = std::log2(vk_memoryRequirements.memoryRequirements.alignment);
    memoryReqs.prefersDedicatedAllocation = vk_dedicatedMemoryRequirements.prefersDedicatedAllocation;
    memoryReqs.requiresDedicatedAllocation = vk_dedicatedMemoryRequirements.requiresDedicatedAllocation;
    return memoryReqs;
}

template<class Interface>
CVulkanDeviceMemoryBacked<Interface>::CVulkanDeviceMemoryBacked(
    const CVulkanLogicalDevice* dev,
    Interface::SCreationParams&& _creationParams,
    const IDeviceMemoryBacked::SDeviceMemoryRequirements& _memReqs,
    const VkResource_t vkHandle
) : Interface(core::smart_refctd_ptr<const ILogicalDevice>(dev),std::move(_creationParams),_memReqs), m_handle(vkHandle)
{
    assert(vkHandle!=VK_NULL_HANDLE);
}

template CVulkanDeviceMemoryBacked<IGPUBuffer>;
template CVulkanDeviceMemoryBacked<IGPUImage>;

}