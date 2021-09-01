#include "CVulkanLogicalDevice.h"
#include "nbl/video/CVulkanPhysicalDevice.h"

namespace nbl::video
{

core::smart_refctd_ptr<IDriverMemoryAllocation> CVulkanLogicalDevice::allocateDeviceLocalMemory(
    const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs)
{
    if (m_physicalDevice->getAPIType() != EAT_VULKAN)
        return nullptr;

    IDriverMemoryBacked::SDriverMemoryRequirements reqs = getDeviceLocalGPUMemoryReqs();
    reqs.vulkanReqs.alignment = additionalReqs.vulkanReqs.alignment;
    reqs.vulkanReqs.size = additionalReqs.vulkanReqs.size;
    reqs.vulkanReqs.memoryTypeBits = additionalReqs.vulkanReqs.memoryTypeBits;

    if (additionalReqs.memoryHeapLocation != IDriverMemoryAllocation::ESMT_DEVICE_LOCAL)
        return nullptr;

    reqs.mappingCapability = additionalReqs.mappingCapability;
    reqs.prefersDedicatedAllocation = additionalReqs.prefersDedicatedAllocation;
    reqs.requiresDedicatedAllocation = additionalReqs.requiresDedicatedAllocation;

    VkMemoryPropertyFlags desiredMemoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    if ((reqs.mappingCapability & IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_READ) ||
        (reqs.mappingCapability & IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_WRITE))
        desiredMemoryProperties |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

    if (reqs.mappingCapability & IDriverMemoryAllocation::EMCF_COHERENT)
        desiredMemoryProperties |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    if (reqs.mappingCapability & IDriverMemoryAllocation::EMCF_CACHED)
        desiredMemoryProperties |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

    const IPhysicalDevice::SMemoryProperties& memoryProperties = m_physicalDevice->getMemoryProperties();

    uint32_t compatibleMemoryTypeCount = 0u;
    uint32_t compatibleMemoryTypeIndices[VK_MAX_MEMORY_TYPES];
    for (uint32_t i = 0u; i < memoryProperties.memoryTypeCount; ++i)
    {
        const bool memoryTypeSupportedForResource = (reqs.vulkanReqs.memoryTypeBits & (1 << i));

        const bool memoryHasDesirableProperties = (memoryProperties.memoryTypes[i].propertyFlags
                & desiredMemoryProperties) == desiredMemoryProperties;

        if (memoryTypeSupportedForResource && memoryHasDesirableProperties)
            compatibleMemoryTypeIndices[compatibleMemoryTypeCount++] = i;
    }

    for (uint32_t i = 0u; i < compatibleMemoryTypeCount; ++i)
    {
        VkMemoryAllocateInfo vk_allocateInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        vk_allocateInfo.pNext = nullptr; // No extensions for now
        vk_allocateInfo.allocationSize = additionalReqs.vulkanReqs.size;
        vk_allocateInfo.memoryTypeIndex = compatibleMemoryTypeIndices[i];

        VkDeviceMemory vk_deviceMemory;
        if (vkAllocateMemory(m_vkdev, &vk_allocateInfo, nullptr, &vk_deviceMemory) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<CVulkanMemoryAllocation>(this, vk_deviceMemory);
        }
    }

    return nullptr;
}

}