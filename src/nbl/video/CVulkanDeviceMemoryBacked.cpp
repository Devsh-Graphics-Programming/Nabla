#define _NBL_VIDEO_C_VULKAN_DEVICE_MEMORY_BACKED_CPP_
#include "nbl/video/CVulkanDeviceMemoryBacked.h"
#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

template<class Interface>
static IDeviceMemoryBacked::SDeviceMemoryRequirements CVulkanDeviceMemoryBacked::obtainRequirements(const ILogicalDevice* device, const VkResouce_t vkHandle)
{
	assert(device->getAPI()==EAT_VULKAN);
    const auto* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(device);
    
    const std::conditional_t<IsImage,VkImageMemoryRequirementsInfo2,VkBufferMemoryRequirementsInfo2> vk_memoryRequirementsInfo = {
        IsImage ? VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2:VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,nullptr,vkHandle
    };

    VkMemoryDedicatedRequirements vk_dedicatedMemoryRequirements = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,nullptr };
    VkMemoryRequirements2 vk_memoryRequirements = { VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,&vk_dedicatedMemoryRequirements };
    const auto& vk = vulkanDevice->getFunctionTable()->vk;
    if constexpr(IsImage)
        vk.vkGetImageMemoryRequirements2(vulkanDevice->getInternalObject(),&vk_memoryRequirementsInfo,&vk_memoryRequirements);
    else
        vk.vkGetBufferMemoryRequirements2(vulkanDevice->getInternalObject(),&vk_memoryRequirementsInfo,&vk_memoryRequirements);

    IDeviceMemoryBacked::SDeviceMemoryRequirements memoryReqs = {};
    memoryReqs.size = vk_memoryRequirements.memoryRequirements.size;
    memoryReqs.memoryTypeBits = vk_memoryRequirements.memoryRequirements.memoryTypeBits;
    memoryReqs.alignmentLog2 = std::log2(vk_memoryRequirements.memoryRequirements.alignment);
    memoryReqs.prefersDedicatedAllocation = vk_dedicatedMemoryRequirements.prefersDedicatedAllocation;
    memoryReqs.requiresDedicatedAllocation = vk_dedicatedMemoryRequirements.requiresDedicatedAllocation;
    return memoryReqs;
}

template CVulkanDeviceMemoryBacked<IGPUBuffer>;
template CVulkanDeviceMemoryBacked<IGPUImage>;

}