#include "CVulkanLogicalDevice.h"

#include "nbl/video/CVulkanPhysicalDevice.h"

namespace nbl::video
{

core::smart_refctd_ptr<IDriverMemoryAllocation> CVulkanLogicalDevice::allocateDeviceLocalMemory(
    const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs)
{
    IDriverMemoryBacked::SDriverMemoryRequirements memoryReqs = getDeviceLocalGPUMemoryReqs();
    memoryReqs.vulkanReqs.alignment = core::max(memoryReqs.vulkanReqs.alignment, additionalReqs.vulkanReqs.alignment);
    memoryReqs.vulkanReqs.size = core::max(memoryReqs.vulkanReqs.size, additionalReqs.vulkanReqs.size);
    memoryReqs.vulkanReqs.memoryTypeBits &= additionalReqs.vulkanReqs.memoryTypeBits;
    memoryReqs.mappingCapability = additionalReqs.mappingCapability;
    memoryReqs.memoryHeapLocation = IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
    memoryReqs.prefersDedicatedAllocation = additionalReqs.prefersDedicatedAllocation;
    memoryReqs.requiresDedicatedAllocation = additionalReqs.requiresDedicatedAllocation;

    return allocateGPUMemory(additionalReqs);
}

core::smart_refctd_ptr<IDriverMemoryAllocation> CVulkanLogicalDevice::allocateSpilloverMemory(
    const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs)
{
    if (additionalReqs.memoryHeapLocation == IDriverMemoryAllocation::ESMT_DEVICE_LOCAL)
        return nullptr;

    IDriverMemoryBacked::SDriverMemoryRequirements memoryReqs = getSpilloverGPUMemoryReqs();
    memoryReqs.vulkanReqs.alignment = core::max(memoryReqs.vulkanReqs.alignment, additionalReqs.vulkanReqs.alignment);
    memoryReqs.vulkanReqs.size = core::max(memoryReqs.vulkanReqs.size, additionalReqs.vulkanReqs.size);
    memoryReqs.vulkanReqs.memoryTypeBits &= additionalReqs.vulkanReqs.memoryTypeBits;
    memoryReqs.mappingCapability = additionalReqs.mappingCapability;
    memoryReqs.memoryHeapLocation = additionalReqs.memoryHeapLocation;
    memoryReqs.prefersDedicatedAllocation = additionalReqs.prefersDedicatedAllocation;
    memoryReqs.requiresDedicatedAllocation = additionalReqs.requiresDedicatedAllocation;

    return allocateGPUMemory(memoryReqs);
}

core::smart_refctd_ptr<IDriverMemoryAllocation> CVulkanLogicalDevice::allocateUpStreamingMemory(
    const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs)
{
    if (getUpStreamingMemoryReqs().memoryHeapLocation != additionalReqs.memoryHeapLocation)
        return nullptr;

    IDriverMemoryBacked::SDriverMemoryRequirements memoryReqs = getUpStreamingMemoryReqs();
    memoryReqs.vulkanReqs.alignment = core::max(memoryReqs.vulkanReqs.alignment, additionalReqs.vulkanReqs.alignment);
    memoryReqs.vulkanReqs.size = core::max(memoryReqs.vulkanReqs.size, additionalReqs.vulkanReqs.size);
    memoryReqs.vulkanReqs.memoryTypeBits &= additionalReqs.vulkanReqs.memoryTypeBits;
    memoryReqs.mappingCapability = additionalReqs.mappingCapability;
    memoryReqs.memoryHeapLocation = additionalReqs.memoryHeapLocation;
    memoryReqs.prefersDedicatedAllocation = additionalReqs.prefersDedicatedAllocation;
    memoryReqs.requiresDedicatedAllocation = additionalReqs.requiresDedicatedAllocation;

    return allocateGPUMemory(memoryReqs);
}

core::smart_refctd_ptr<IDriverMemoryAllocation> CVulkanLogicalDevice::allocateDownStreamingMemory(
    const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs)
{
    if (getDownStreamingMemoryReqs().memoryHeapLocation != additionalReqs.memoryHeapLocation)
        return nullptr;

    IDriverMemoryBacked::SDriverMemoryRequirements memoryReqs = getDownStreamingMemoryReqs();
    memoryReqs.vulkanReqs.alignment = core::max(memoryReqs.vulkanReqs.alignment, additionalReqs.vulkanReqs.alignment);
    memoryReqs.vulkanReqs.size = core::max(memoryReqs.vulkanReqs.size, additionalReqs.vulkanReqs.size);
    memoryReqs.vulkanReqs.memoryTypeBits &= additionalReqs.vulkanReqs.memoryTypeBits;
    memoryReqs.mappingCapability = additionalReqs.mappingCapability;
    memoryReqs.memoryHeapLocation = additionalReqs.memoryHeapLocation;
    memoryReqs.prefersDedicatedAllocation = additionalReqs.prefersDedicatedAllocation;
    memoryReqs.requiresDedicatedAllocation = additionalReqs.requiresDedicatedAllocation;

    return allocateGPUMemory(memoryReqs);
}

core::smart_refctd_ptr<IDriverMemoryAllocation> CVulkanLogicalDevice::allocateCPUSideGPUVisibleMemory(
    const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs)
{
    if (additionalReqs.memoryHeapLocation != IDriverMemoryAllocation::ESMT_NOT_DEVICE_LOCAL)
        return nullptr;

    IDriverMemoryBacked::SDriverMemoryRequirements memoryReqs = getCPUSideGPUVisibleGPUMemoryReqs();
    memoryReqs.vulkanReqs.alignment = core::max(memoryReqs.vulkanReqs.alignment, additionalReqs.vulkanReqs.alignment);
    memoryReqs.vulkanReqs.size = core::max(memoryReqs.vulkanReqs.size, additionalReqs.vulkanReqs.size);
    memoryReqs.vulkanReqs.memoryTypeBits &= additionalReqs.vulkanReqs.memoryTypeBits;
    memoryReqs.mappingCapability = additionalReqs.mappingCapability;
    memoryReqs.memoryHeapLocation = additionalReqs.memoryHeapLocation;
    memoryReqs.prefersDedicatedAllocation = additionalReqs.prefersDedicatedAllocation;
    memoryReqs.requiresDedicatedAllocation = additionalReqs.requiresDedicatedAllocation;

    return allocateGPUMemory(memoryReqs);
}

core::smart_refctd_ptr<IDriverMemoryAllocation> CVulkanLogicalDevice::allocateGPUMemory(
    const IDriverMemoryBacked::SDriverMemoryRequirements& reqs)
{
    VkMemoryPropertyFlags desiredMemoryProperties = static_cast<VkMemoryPropertyFlags>(0u);

    if (reqs.memoryHeapLocation == IDriverMemoryAllocation::ESMT_DEVICE_LOCAL)
        desiredMemoryProperties |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

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
        // Todo(achal): Make use of requiresDedicatedAllocation and prefersDedicatedAllocation

        VkMemoryAllocateInfo vk_allocateInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        vk_allocateInfo.pNext = nullptr; // No extensions for now
        vk_allocateInfo.allocationSize = reqs.vulkanReqs.size;
        vk_allocateInfo.memoryTypeIndex = compatibleMemoryTypeIndices[i];

        VkDeviceMemory vk_deviceMemory;
        if (vkAllocateMemory(m_vkdev, &vk_allocateInfo, nullptr, &vk_deviceMemory) == VK_SUCCESS)
        {
            // Todo(achal): Change dedicate to not always be false
            return core::make_smart_refctd_ptr<CVulkanMemoryAllocation>(this, reqs.vulkanReqs.size, false, vk_deviceMemory);
        }
    }

    return nullptr;
}

}