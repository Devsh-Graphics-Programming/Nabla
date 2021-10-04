#include "CVulkanLogicalDevice.h"

#include "nbl/video/CVulkanPhysicalDevice.h"
#include "nbl/video/CVulkanCommandBuffer.h"

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
        if (m_devf.vk.vkAllocateMemory(m_vkdev, &vk_allocateInfo, nullptr, &vk_deviceMemory) == VK_SUCCESS)
        {
            // Todo(achal): Change dedicate to not always be false
            return core::make_smart_refctd_ptr<CVulkanMemoryAllocation>(this, reqs.vulkanReqs.size, false, vk_deviceMemory);
        }
    }

    return nullptr;
}

bool CVulkanLogicalDevice::createCommandBuffers_impl(IGPUCommandPool* cmdPool, IGPUCommandBuffer::E_LEVEL level,
    uint32_t count, core::smart_refctd_ptr<IGPUCommandBuffer>* outCmdBufs)
{
    constexpr uint32_t MAX_COMMAND_BUFFER_COUNT = 1000u;

    if (cmdPool->getAPIType() != EAT_VULKAN)
        return false;

    auto vulkanCommandPool = static_cast<CVulkanCommandPool*>(cmdPool)->getInternalObject();

    assert(count <= MAX_COMMAND_BUFFER_COUNT);
    VkCommandBuffer vk_commandBuffers[MAX_COMMAND_BUFFER_COUNT];

    VkCommandBufferAllocateInfo vk_allocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    vk_allocateInfo.pNext = nullptr; // this must be NULL
    vk_allocateInfo.commandPool = vulkanCommandPool;
    vk_allocateInfo.level = static_cast<VkCommandBufferLevel>(level);
    vk_allocateInfo.commandBufferCount = count;

    if (m_devf.vk.vkAllocateCommandBuffers(m_vkdev, &vk_allocateInfo, vk_commandBuffers) == VK_SUCCESS)
    {
        for (uint32_t i = 0u; i < count; ++i)
        {
            outCmdBufs[i] = core::make_smart_refctd_ptr<CVulkanCommandBuffer>(
                core::smart_refctd_ptr<ILogicalDevice>(this), level, vk_commandBuffers[i],
                cmdPool);
        }

        return true;
    }
    else
    {
        return false;
    }
}

core::smart_refctd_ptr<IGPUImage> CVulkanLogicalDevice::createGPUImage(asset::IImage::SCreationParams&& params)
{
    VkImageCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    vk_createInfo.pNext = nullptr; // there are a lot of extensions
    vk_createInfo.flags = static_cast<VkImageCreateFlags>(params.flags);
    vk_createInfo.imageType = static_cast<VkImageType>(params.type);
    vk_createInfo.format = getVkFormatFromFormat(params.format);
    vk_createInfo.extent = { params.extent.width, params.extent.height, params.extent.depth };
    vk_createInfo.mipLevels = params.mipLevels;
    vk_createInfo.arrayLayers = params.arrayLayers;
    vk_createInfo.samples = static_cast<VkSampleCountFlagBits>(params.samples);
    vk_createInfo.tiling = static_cast<VkImageTiling>(params.tiling);
    vk_createInfo.usage = static_cast<VkImageUsageFlags>(params.usage.value);
    vk_createInfo.sharingMode = static_cast<VkSharingMode>(params.sharingMode);
    vk_createInfo.queueFamilyIndexCount = params.queueFamilyIndexCount;
    vk_createInfo.pQueueFamilyIndices = params.queueFamilyIndices;
    vk_createInfo.initialLayout = static_cast<VkImageLayout>(params.initialLayout);

    VkImage vk_image;
    if (m_devf.vk.vkCreateImage(m_vkdev, &vk_createInfo, nullptr, &vk_image) == VK_SUCCESS)
    {
        VkImageMemoryRequirementsInfo2 vk_memReqsInfo = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2 };
        vk_memReqsInfo.pNext = nullptr;
        vk_memReqsInfo.image = vk_image;

        VkMemoryDedicatedRequirements vk_memDedReqs = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS };
        VkMemoryRequirements2 vk_memReqs = { VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2 };
        vk_memReqs.pNext = &vk_memDedReqs;

        m_devf.vk.vkGetImageMemoryRequirements2(m_vkdev, &vk_memReqsInfo, &vk_memReqs);

        IDriverMemoryBacked::SDriverMemoryRequirements imageMemReqs = {};
        imageMemReqs.vulkanReqs.alignment = vk_memReqs.memoryRequirements.alignment;
        imageMemReqs.vulkanReqs.size = vk_memReqs.memoryRequirements.size;
        imageMemReqs.vulkanReqs.memoryTypeBits = vk_memReqs.memoryRequirements.memoryTypeBits;
        imageMemReqs.memoryHeapLocation = 0u; // doesn't matter, would get overwritten during memory allocation for this resource anyway
        imageMemReqs.mappingCapability = 0u; // doesn't matter, would get overwritten during memory allocation for this resource anyway
        imageMemReqs.prefersDedicatedAllocation = vk_memDedReqs.prefersDedicatedAllocation;
        imageMemReqs.requiresDedicatedAllocation = vk_memDedReqs.requiresDedicatedAllocation;

        return core::make_smart_refctd_ptr<CVulkanImage>(
            core::smart_refctd_ptr<CVulkanLogicalDevice>(this), std::move(params),
            vk_image, imageMemReqs);
    }
    else
    {
        return nullptr;
    }
}

}