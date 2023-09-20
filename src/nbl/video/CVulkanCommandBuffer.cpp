#include "nbl/video/CVulkanCommandBuffer.h"

#include "nbl/video/CVulkanCommon.h"
#include "nbl/video/CVulkanLogicalDevice.h"
#include "nbl/video/CVulkanQueryPool.h"

namespace nbl::video
{

bool CVulkanCommandBuffer::begin_impl(core::bitflag<E_USAGE> recordingFlags, const SInheritanceInfo* inheritanceInfo)
{
    VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.pNext = nullptr; // pNext must be NULL or a pointer to a valid instance of VkDeviceGroupCommandBufferBeginInfo
    beginInfo.flags = static_cast<VkCommandBufferUsageFlags>(recordingFlags.value);

    VkCommandBufferInheritanceInfo vk_inheritanceInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO };
    if (inheritanceInfo)
    {
        vk_inheritanceInfo.renderPass = IBackendObject::compatibility_cast<const CVulkanRenderpass*>(inheritanceInfo->renderpass.get(), this)->getInternalObject();
        vk_inheritanceInfo.subpass = inheritanceInfo->subpass;
        // From the spec:
        // Specifying the exact framebuffer that the secondary command buffer will be
        // executed with may result in better performance at command buffer execution time.
        if (inheritanceInfo->framebuffer)
            vk_inheritanceInfo.framebuffer = IBackendObject::compatibility_cast<const CVulkanFramebuffer*>(inheritanceInfo->framebuffer.get(), this)->getInternalObject();
        vk_inheritanceInfo.occlusionQueryEnable = inheritanceInfo->occlusionQueryEnable;
        vk_inheritanceInfo.queryFlags = static_cast<VkQueryControlFlags>(inheritanceInfo->queryFlags.value);
        vk_inheritanceInfo.pipelineStatistics = static_cast<VkQueryPipelineStatisticFlags>(0u); // must be 0

        beginInfo.pInheritanceInfo = &vk_inheritanceInfo;
    }

    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    const VkResult retval = vk->vk.vkBeginCommandBuffer(m_cmdbuf, &beginInfo);
    return retval == VK_SUCCESS;
}

bool CVulkanCommandBuffer::setViewport(uint32_t firstViewport, uint32_t viewportCount, const asset::SViewport* pViewports)
{
    constexpr uint32_t MAX_VIEWPORT_COUNT = (1u << 12) / sizeof(VkViewport);
    assert(viewportCount <= MAX_VIEWPORT_COUNT);

    VkViewport vk_viewports[MAX_VIEWPORT_COUNT];
    for (uint32_t i = 0u; i < viewportCount; ++i)
    {
        vk_viewports[i].x = pViewports[i].x;
        vk_viewports[i].y = pViewports[i].y;
        vk_viewports[i].width = pViewports[i].width;
        vk_viewports[i].height = pViewports[i].height;
        vk_viewports[i].minDepth = pViewports[i].minDepth;
        vk_viewports[i].maxDepth = pViewports[i].maxDepth;
    }

    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    vk->vk.vkCmdSetViewport(m_cmdbuf, firstViewport, viewportCount, vk_viewports);
    return true;
}

bool CVulkanCommandBuffer::copyBuffer_impl(const buffer_t* srcBuffer, buffer_t* dstBuffer, uint32_t regionCount, const asset::SBufferCopy* pRegions)
{
    VkBuffer vk_srcBuffer = IBackendObject::compatibility_cast<const CVulkanBuffer*>(srcBuffer, this)->getInternalObject();
    VkBuffer vk_dstBuffer = IBackendObject::compatibility_cast<const CVulkanBuffer*>(dstBuffer, this)->getInternalObject();

    constexpr uint32_t MAX_BUFFER_COPY_REGION_COUNT = 681u;
    assert(regionCount <= MAX_BUFFER_COPY_REGION_COUNT);
    VkBufferCopy vk_bufferCopyRegions[MAX_BUFFER_COPY_REGION_COUNT];
    for (uint32_t i = 0u; i < regionCount; ++i)
    {
        vk_bufferCopyRegions[i].srcOffset = pRegions[i].srcOffset;
        vk_bufferCopyRegions[i].dstOffset = pRegions[i].dstOffset;
        vk_bufferCopyRegions[i].size = pRegions[i].size;
    }

    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    vk->vk.vkCmdCopyBuffer(m_cmdbuf, vk_srcBuffer, vk_dstBuffer, regionCount, vk_bufferCopyRegions);

    return true;
}

bool CVulkanCommandBuffer::copyImage_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SImageCopy* pRegions)
{
    constexpr uint32_t MAX_COUNT = (1u << 12) / sizeof(VkImageCopy);
    assert(regionCount <= MAX_COUNT);

    VkImageCopy vk_regions[MAX_COUNT];
    for (uint32_t i = 0u; i < regionCount; ++i)
    {
        vk_regions[i].srcSubresource.aspectMask = static_cast<VkImageAspectFlags>(pRegions[i].srcSubresource.aspectMask.value);
        vk_regions[i].srcSubresource.baseArrayLayer = pRegions[i].srcSubresource.baseArrayLayer;
        vk_regions[i].srcSubresource.layerCount = pRegions[i].srcSubresource.layerCount;
        vk_regions[i].srcSubresource.mipLevel = pRegions[i].srcSubresource.mipLevel;

        vk_regions[i].srcOffset = { static_cast<int32_t>(pRegions[i].srcOffset.x), static_cast<int32_t>(pRegions[i].srcOffset.y), static_cast<int32_t>(pRegions[i].srcOffset.z) };

        vk_regions[i].dstSubresource.aspectMask = static_cast<VkImageAspectFlags>(pRegions[i].dstSubresource.aspectMask.value);
        vk_regions[i].dstSubresource.baseArrayLayer = pRegions[i].dstSubresource.baseArrayLayer;
        vk_regions[i].dstSubresource.layerCount = pRegions[i].dstSubresource.layerCount;
        vk_regions[i].dstSubresource.mipLevel = pRegions[i].dstSubresource.mipLevel;

        vk_regions[i].dstOffset = { static_cast<int32_t>(pRegions[i].dstOffset.x), static_cast<int32_t>(pRegions[i].dstOffset.y), static_cast<int32_t>(pRegions[i].dstOffset.z) };

        vk_regions[i].extent = { pRegions[i].extent.width, pRegions[i].extent.height, pRegions[i].extent.depth };
    }

    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    vk->vk.vkCmdCopyImage(
        m_cmdbuf,
        IBackendObject::compatibility_cast<const CVulkanImage*>(srcImage, this)->getInternalObject(),
        static_cast<VkImageLayout>(srcImageLayout),
        IBackendObject::compatibility_cast<const CVulkanImage*>(dstImage, this)->getInternalObject(),
        static_cast<VkImageLayout>(dstImageLayout),
        regionCount,
        vk_regions);

    return true;
}

bool CVulkanCommandBuffer::copyBufferToImage_impl(const buffer_t* srcBuffer, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions)
{
    constexpr uint32_t MAX_REGION_COUNT = (1ull << 12) / sizeof(VkBufferImageCopy);
    assert(regionCount <= MAX_REGION_COUNT);

    VkBufferImageCopy vk_regions[MAX_REGION_COUNT];
    for (uint32_t i = 0u; i < regionCount; ++i)
    {
        vk_regions[i].bufferOffset = pRegions[i].bufferOffset;
        vk_regions[i].bufferRowLength = pRegions[i].bufferRowLength;
        vk_regions[i].bufferImageHeight = pRegions[i].bufferImageHeight;
        vk_regions[i].imageSubresource.aspectMask = static_cast<VkImageAspectFlags>(pRegions[i].imageSubresource.aspectMask.value);
        vk_regions[i].imageSubresource.mipLevel = pRegions[i].imageSubresource.mipLevel;
        vk_regions[i].imageSubresource.baseArrayLayer = pRegions[i].imageSubresource.baseArrayLayer;
        vk_regions[i].imageSubresource.layerCount = pRegions[i].imageSubresource.layerCount;
        vk_regions[i].imageOffset = { static_cast<int32_t>(pRegions[i].imageOffset.x), static_cast<int32_t>(pRegions[i].imageOffset.y), static_cast<int32_t>(pRegions[i].imageOffset.z) }; // Todo(achal): Make the regular old assignment operator work
        vk_regions[i].imageExtent = { pRegions[i].imageExtent.width, pRegions[i].imageExtent.height, pRegions[i].imageExtent.depth }; // Todo(achal): Make the regular old assignment operator work
    }

    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    vk->vk.vkCmdCopyBufferToImage(m_cmdbuf,
        IBackendObject::compatibility_cast<const video::CVulkanBuffer*>(srcBuffer, this)->getInternalObject(),
        IBackendObject::compatibility_cast<const video::CVulkanImage*>(dstImage, this)->getInternalObject(),
        static_cast<VkImageLayout>(dstImageLayout), regionCount, vk_regions);

    return true;
}

bool CVulkanCommandBuffer::copyImageToBuffer_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, buffer_t* dstBuffer, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions)
{
    VkImage vk_srcImage = IBackendObject::compatibility_cast<const CVulkanImage*>(srcImage, this)->getInternalObject();
    VkBuffer vk_dstBuffer = IBackendObject::compatibility_cast<const CVulkanBuffer*>(dstBuffer, this)->getInternalObject();

    constexpr uint32_t MAX_REGION_COUNT = (1u << 12) / sizeof(VkBufferImageCopy);
    VkBufferImageCopy vk_copyRegions[MAX_REGION_COUNT];
    assert(regionCount <= MAX_REGION_COUNT);

    for (uint32_t i = 0u; i < regionCount; ++i)
    {
        vk_copyRegions[i].bufferOffset = static_cast<VkDeviceSize>(pRegions[i].bufferOffset);
        vk_copyRegions[i].bufferRowLength = pRegions[i].bufferRowLength;
        vk_copyRegions[i].bufferImageHeight = pRegions[i].bufferImageHeight;
        vk_copyRegions[i].imageSubresource.aspectMask = static_cast<VkImageAspectFlags>(pRegions[i].imageSubresource.aspectMask.value);
        vk_copyRegions[i].imageSubresource.baseArrayLayer = pRegions[i].imageSubresource.baseArrayLayer;
        vk_copyRegions[i].imageSubresource.layerCount = pRegions[i].imageSubresource.layerCount;
        vk_copyRegions[i].imageSubresource.mipLevel = pRegions[i].imageSubresource.mipLevel;
        vk_copyRegions[i].imageOffset = { static_cast<int32_t>(pRegions[i].imageOffset.x), static_cast<int32_t>(pRegions[i].imageOffset.y), static_cast<int32_t>(pRegions[i].imageOffset.z) };
        vk_copyRegions[i].imageExtent = { pRegions[i].imageExtent.width, pRegions[i].imageExtent.height, pRegions[i].imageExtent.depth };
    }

    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    vk->vk.vkCmdCopyImageToBuffer(
        m_cmdbuf,
        vk_srcImage,
        static_cast<VkImageLayout>(srcImageLayout),
        vk_dstBuffer,
        regionCount,
        vk_copyRegions);

    return true;
}

bool CVulkanCommandBuffer::blitImage_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageBlit* pRegions, asset::ISampler::E_TEXTURE_FILTER filter)
{
    VkImage vk_srcImage = IBackendObject::compatibility_cast<const CVulkanImage*>(srcImage, this)->getInternalObject();
    VkImage vk_dstImage = IBackendObject::compatibility_cast<const CVulkanImage*>(dstImage, this)->getInternalObject();

    constexpr uint32_t MAX_BLIT_REGION_COUNT = 100u;
    VkImageBlit vk_blitRegions[MAX_BLIT_REGION_COUNT];
    assert(regionCount <= MAX_BLIT_REGION_COUNT);

    for (uint32_t i = 0u; i < regionCount; ++i)
    {
        vk_blitRegions[i].srcSubresource.aspectMask = static_cast<VkImageAspectFlags>(pRegions[i].srcSubresource.aspectMask.value);
        vk_blitRegions[i].srcSubresource.mipLevel = pRegions[i].srcSubresource.mipLevel;
        vk_blitRegions[i].srcSubresource.baseArrayLayer = pRegions[i].srcSubresource.baseArrayLayer;
        vk_blitRegions[i].srcSubresource.layerCount = pRegions[i].srcSubresource.layerCount;

        // Todo(achal): Remove `static_cast`s
        vk_blitRegions[i].srcOffsets[0] = { static_cast<int32_t>(pRegions[i].srcOffsets[0].x), static_cast<int32_t>(pRegions[i].srcOffsets[0].y), static_cast<int32_t>(pRegions[i].srcOffsets[0].z) };
        vk_blitRegions[i].srcOffsets[1] = { static_cast<int32_t>(pRegions[i].srcOffsets[1].x), static_cast<int32_t>(pRegions[i].srcOffsets[1].y), static_cast<int32_t>(pRegions[i].srcOffsets[1].z) };

        vk_blitRegions[i].dstSubresource.aspectMask = static_cast<VkImageAspectFlags>(pRegions[i].dstSubresource.aspectMask.value);
        vk_blitRegions[i].dstSubresource.mipLevel = pRegions[i].dstSubresource.mipLevel;
        vk_blitRegions[i].dstSubresource.baseArrayLayer = pRegions[i].dstSubresource.baseArrayLayer;
        vk_blitRegions[i].dstSubresource.layerCount = pRegions[i].dstSubresource.layerCount;

        // Todo(achal): Remove `static_cast`s
        vk_blitRegions[i].dstOffsets[0] = { static_cast<int32_t>(pRegions[i].dstOffsets[0].x), static_cast<int32_t>(pRegions[i].dstOffsets[0].y), static_cast<int32_t>(pRegions[i].dstOffsets[0].z) };
        vk_blitRegions[i].dstOffsets[1] = { static_cast<int32_t>(pRegions[i].dstOffsets[1].x), static_cast<int32_t>(pRegions[i].dstOffsets[1].y), static_cast<int32_t>(pRegions[i].dstOffsets[1].z) };
    }

    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    vk->vk.vkCmdBlitImage(m_cmdbuf, vk_srcImage, static_cast<VkImageLayout>(srcImageLayout),
        vk_dstImage, static_cast<VkImageLayout>(dstImageLayout), regionCount, vk_blitRegions,
        static_cast<VkFilter>(filter));

    return true;
}

bool CVulkanCommandBuffer::resolveImage_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageResolve* pRegions)
{
    constexpr uint32_t MAX_COUNT = (1u << 12) / sizeof(VkImageResolve);
    assert(regionCount <= MAX_COUNT);

    VkImageResolve vk_regions[MAX_COUNT];
    for (uint32_t i = 0u; i < regionCount; ++i)
    {
        vk_regions[i].srcSubresource.aspectMask = static_cast<VkImageAspectFlags>(pRegions[i].srcSubresource.aspectMask.value);
        vk_regions[i].srcSubresource.baseArrayLayer = pRegions[i].srcSubresource.baseArrayLayer;
        vk_regions[i].srcSubresource.layerCount = pRegions[i].srcSubresource.layerCount;
        vk_regions[i].srcSubresource.mipLevel = pRegions[i].srcSubresource.mipLevel;

        vk_regions[i].srcOffset = { static_cast<int32_t>(pRegions[i].srcOffset.x), static_cast<int32_t>(pRegions[i].srcOffset.y), static_cast<int32_t>(pRegions[i].srcOffset.z) };

        vk_regions[i].dstSubresource.aspectMask = static_cast<VkImageAspectFlags>(pRegions[i].dstSubresource.aspectMask.value);
        vk_regions[i].dstSubresource.baseArrayLayer = pRegions[i].dstSubresource.baseArrayLayer;
        vk_regions[i].dstSubresource.layerCount = pRegions[i].dstSubresource.layerCount;
        vk_regions[i].dstSubresource.mipLevel = pRegions[i].dstSubresource.mipLevel;

        vk_regions[i].dstOffset = { static_cast<int32_t>(pRegions[i].dstOffset.x), static_cast<int32_t>(pRegions[i].dstOffset.y), static_cast<int32_t>(pRegions[i].dstOffset.z) };

        vk_regions[i].extent = { pRegions[i].extent.width, pRegions[i].extent.height, pRegions[i].extent.depth };
    }

    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    vk->vk.vkCmdResolveImage(
        m_cmdbuf,
        IBackendObject::compatibility_cast<const CVulkanImage*>(srcImage, this)->getInternalObject(),
        static_cast<VkImageLayout>(srcImageLayout),
        IBackendObject::compatibility_cast<const CVulkanImage*>(dstImage, this)->getInternalObject(),
        static_cast<VkImageLayout>(dstImageLayout),
        regionCount,
        vk_regions);

    return true;
}

void CVulkanCommandBuffer::bindVertexBuffers_impl(uint32_t firstBinding, uint32_t bindingCount, const buffer_t* const* const pBuffers, const size_t* pOffsets)
{
    constexpr uint32_t MaxBufferCount = asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT;
    assert(bindingCount <= MaxBufferCount);

    VkBuffer vk_buffers[MaxBufferCount];
    VkDeviceSize vk_offsets[MaxBufferCount];

    VkBuffer dummyBuffer = VK_NULL_HANDLE;
    for (uint32_t i = 0u; i < bindingCount; ++i)
    {
        if (!pBuffers[i] || (pBuffers[i]->getAPIType() != EAT_VULKAN))
        {
            vk_buffers[i] = dummyBuffer;
            vk_offsets[i] = 0;
        }
        else
        {
            VkBuffer vk_buffer = IBackendObject::compatibility_cast<const CVulkanBuffer*>(pBuffers[i], this)->getInternalObject();
            if (dummyBuffer == VK_NULL_HANDLE)
                dummyBuffer = vk_buffer;

            vk_buffers[i] = vk_buffer;
            vk_offsets[i] = static_cast<VkDeviceSize>(pOffsets[i]);
        }
    }
    for (uint32_t i = 0u; i < bindingCount; ++i)
    {
        if (vk_buffers[i] == VK_NULL_HANDLE)
            vk_buffers[i] = dummyBuffer;
    }

    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    vk->vk.vkCmdBindVertexBuffers(
        m_cmdbuf,
        firstBinding,
        bindingCount,
        vk_buffers,
        vk_offsets);
}

bool CVulkanCommandBuffer::waitEvents_impl(uint32_t eventCount, event_t* const* const pEvents, const SDependencyInfo* depInfo)
{
    constexpr uint32_t MAX_EVENT_COUNT = (1u << 12) / sizeof(VkEvent);
    assert(eventCount <= MAX_EVENT_COUNT);

    constexpr uint32_t MAX_BARRIER_COUNT = 100u;
    assert(depInfo->memBarrierCount <= MAX_BARRIER_COUNT);
    assert(depInfo->bufBarrierCount <= MAX_BARRIER_COUNT);
    assert(depInfo->imgBarrierCount <= MAX_BARRIER_COUNT);

    VkEvent vk_events[MAX_EVENT_COUNT];
    for (uint32_t i = 0u; i < eventCount; ++i)
        vk_events[i] = IBackendObject::compatibility_cast<const CVulkanEvent*>(pEvents[i], this)->getInternalObject();

    VkMemoryBarrier vk_memoryBarriers[MAX_BARRIER_COUNT];
    for (uint32_t i = 0u; i < depInfo->memBarrierCount; ++i)
    {
        vk_memoryBarriers[i] = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        vk_memoryBarriers[i].pNext = nullptr; // must be NULL
        vk_memoryBarriers[i].srcAccessMask = static_cast<VkAccessFlags>(depInfo->memBarriers[i].srcAccessMask.value);
        vk_memoryBarriers[i].dstAccessMask = static_cast<VkAccessFlags>(depInfo->memBarriers[i].dstAccessMask.value);
    }

    VkBufferMemoryBarrier vk_bufferMemoryBarriers[MAX_BARRIER_COUNT];
    for (uint32_t i = 0u; i < depInfo->bufBarrierCount; ++i)
    {
        vk_bufferMemoryBarriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        vk_bufferMemoryBarriers[i].pNext = nullptr; // must be NULL
        vk_bufferMemoryBarriers[i].srcAccessMask = static_cast<VkAccessFlags>(depInfo->bufBarriers[i].barrier.srcAccessMask.value);
        vk_bufferMemoryBarriers[i].dstAccessMask = static_cast<VkAccessFlags>(depInfo->bufBarriers[i].barrier.dstAccessMask.value);
        vk_bufferMemoryBarriers[i].srcQueueFamilyIndex = depInfo->bufBarriers[i].srcQueueFamilyIndex;
        vk_bufferMemoryBarriers[i].dstQueueFamilyIndex = depInfo->bufBarriers[i].dstQueueFamilyIndex;
        vk_bufferMemoryBarriers[i].buffer = IBackendObject::compatibility_cast<const CVulkanBuffer*>(depInfo->bufBarriers[i].buffer.get(), this)->getInternalObject();
        vk_bufferMemoryBarriers[i].offset = depInfo->bufBarriers[i].offset;
        vk_bufferMemoryBarriers[i].size = depInfo->bufBarriers[i].size;
    }

    VkImageMemoryBarrier vk_imageMemoryBarriers[MAX_BARRIER_COUNT];
    for (uint32_t i = 0u; i < depInfo->imgBarrierCount; ++i)
    {
        vk_imageMemoryBarriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        vk_imageMemoryBarriers[i].pNext = nullptr; // pNext must be NULL or a pointer to a valid instance of VkSampleLocationsInfoEXT
        vk_imageMemoryBarriers[i].srcAccessMask = static_cast<VkAccessFlags>(depInfo->imgBarriers[i].barrier.srcAccessMask.value);
        vk_imageMemoryBarriers[i].dstAccessMask = static_cast<VkAccessFlags>(depInfo->imgBarriers[i].barrier.dstAccessMask.value);
        vk_imageMemoryBarriers[i].oldLayout = static_cast<VkImageLayout>(depInfo->imgBarriers[i].oldLayout);
        vk_imageMemoryBarriers[i].newLayout = static_cast<VkImageLayout>(depInfo->imgBarriers[i].newLayout);
        vk_imageMemoryBarriers[i].srcQueueFamilyIndex = depInfo->imgBarriers[i].srcQueueFamilyIndex;
        vk_imageMemoryBarriers[i].dstQueueFamilyIndex = depInfo->imgBarriers[i].dstQueueFamilyIndex;
        vk_imageMemoryBarriers[i].image = IBackendObject::compatibility_cast<const CVulkanImage*>(depInfo->imgBarriers[i].image.get(), this)->getInternalObject();
        vk_imageMemoryBarriers[i].subresourceRange.aspectMask = static_cast<VkImageAspectFlags>(depInfo->imgBarriers[i].subresourceRange.aspectMask.value);
        vk_imageMemoryBarriers[i].subresourceRange.baseMipLevel = depInfo->imgBarriers[i].subresourceRange.baseMipLevel;
        vk_imageMemoryBarriers[i].subresourceRange.levelCount = depInfo->imgBarriers[i].subresourceRange.levelCount;
        vk_imageMemoryBarriers[i].subresourceRange.baseArrayLayer = depInfo->imgBarriers[i].subresourceRange.baseArrayLayer;
        vk_imageMemoryBarriers[i].subresourceRange.layerCount = depInfo->imgBarriers[i].subresourceRange.layerCount;
    }

    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    vk->vk.vkCmdWaitEvents(
        m_cmdbuf,
        eventCount,
        vk_events,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, // No way to get this!
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, // No way to get this!
        depInfo->memBarrierCount,
        vk_memoryBarriers,
        depInfo->bufBarrierCount,
        vk_bufferMemoryBarriers,
        depInfo->imgBarrierCount,
        vk_imageMemoryBarriers);

    return true;
}

bool CVulkanCommandBuffer::pipelineBarrier_impl(core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> srcStageMask,
    core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> dstStageMask,
    core::bitflag<asset::E_DEPENDENCY_FLAGS> dependencyFlags,
    uint32_t memoryBarrierCount, const asset::SMemoryBarrier* pMemoryBarriers,
    uint32_t bufferMemoryBarrierCount, const SBufferMemoryBarrier* pBufferMemoryBarriers,
    uint32_t imageMemoryBarrierCount, const SImageMemoryBarrier* pImageMemoryBarriers)
{
    constexpr uint32_t MAX_BARRIER_COUNT = 100u;

    assert(memoryBarrierCount <= MAX_BARRIER_COUNT);
    assert(bufferMemoryBarrierCount <= MAX_BARRIER_COUNT);
    assert(imageMemoryBarrierCount <= MAX_BARRIER_COUNT);

    VkMemoryBarrier vk_memoryBarriers[MAX_BARRIER_COUNT];
    for (uint32_t i = 0u; i < memoryBarrierCount; ++i)
    {
        vk_memoryBarriers[i] = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        vk_memoryBarriers[i].pNext = nullptr; // must be NULL
        vk_memoryBarriers[i].srcAccessMask = static_cast<VkAccessFlags>(pMemoryBarriers[i].srcAccessMask.value);
        vk_memoryBarriers[i].dstAccessMask = static_cast<VkAccessFlags>(pMemoryBarriers[i].dstAccessMask.value);
    }

    VkBufferMemoryBarrier vk_bufferMemoryBarriers[MAX_BARRIER_COUNT];
    for (uint32_t i = 0u; i < bufferMemoryBarrierCount; ++i)
    {
        vk_bufferMemoryBarriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        vk_bufferMemoryBarriers[i].pNext = nullptr; // must be NULL
        vk_bufferMemoryBarriers[i].srcAccessMask = static_cast<VkAccessFlags>(pBufferMemoryBarriers[i].barrier.srcAccessMask.value);
        vk_bufferMemoryBarriers[i].dstAccessMask = static_cast<VkAccessFlags>(pBufferMemoryBarriers[i].barrier.dstAccessMask.value);
        vk_bufferMemoryBarriers[i].srcQueueFamilyIndex = pBufferMemoryBarriers[i].srcQueueFamilyIndex;
        vk_bufferMemoryBarriers[i].dstQueueFamilyIndex = pBufferMemoryBarriers[i].dstQueueFamilyIndex;
        vk_bufferMemoryBarriers[i].buffer = IBackendObject::compatibility_cast<const CVulkanBuffer*>(pBufferMemoryBarriers[i].buffer.get(), this)->getInternalObject();
        vk_bufferMemoryBarriers[i].offset = pBufferMemoryBarriers[i].offset;
        vk_bufferMemoryBarriers[i].size = pBufferMemoryBarriers[i].size;
    }

    VkImageMemoryBarrier vk_imageMemoryBarriers[MAX_BARRIER_COUNT];
    for (uint32_t i = 0u; i < imageMemoryBarrierCount; ++i)
    {
        vk_imageMemoryBarriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        vk_imageMemoryBarriers[i].pNext = nullptr; // pNext must be NULL or a pointer to a valid instance of VkSampleLocationsInfoEXT
        vk_imageMemoryBarriers[i].srcAccessMask = static_cast<VkAccessFlags>(pImageMemoryBarriers[i].barrier.srcAccessMask.value);
        vk_imageMemoryBarriers[i].dstAccessMask = static_cast<VkAccessFlags>(pImageMemoryBarriers[i].barrier.dstAccessMask.value);
        vk_imageMemoryBarriers[i].oldLayout = static_cast<VkImageLayout>(pImageMemoryBarriers[i].oldLayout);
        vk_imageMemoryBarriers[i].newLayout = static_cast<VkImageLayout>(pImageMemoryBarriers[i].newLayout);
        vk_imageMemoryBarriers[i].srcQueueFamilyIndex = pImageMemoryBarriers[i].srcQueueFamilyIndex;
        vk_imageMemoryBarriers[i].dstQueueFamilyIndex = pImageMemoryBarriers[i].dstQueueFamilyIndex;
        vk_imageMemoryBarriers[i].image = IBackendObject::compatibility_cast<const CVulkanImage*>(pImageMemoryBarriers[i].image.get(), this)->getInternalObject();
        vk_imageMemoryBarriers[i].subresourceRange.aspectMask = static_cast<VkImageAspectFlags>(pImageMemoryBarriers[i].subresourceRange.aspectMask.value);
        vk_imageMemoryBarriers[i].subresourceRange.baseMipLevel = pImageMemoryBarriers[i].subresourceRange.baseMipLevel;
        vk_imageMemoryBarriers[i].subresourceRange.levelCount = pImageMemoryBarriers[i].subresourceRange.levelCount;
        vk_imageMemoryBarriers[i].subresourceRange.baseArrayLayer = pImageMemoryBarriers[i].subresourceRange.baseArrayLayer;
        vk_imageMemoryBarriers[i].subresourceRange.layerCount = pImageMemoryBarriers[i].subresourceRange.layerCount;
    }

    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    vk->vk.vkCmdPipelineBarrier(m_cmdbuf, getVkPipelineStageFlagsFromPipelineStageFlags(srcStageMask.value),
        getVkPipelineStageFlagsFromPipelineStageFlags(dstStageMask.value),
        static_cast<VkDependencyFlags>(dependencyFlags.value),
        memoryBarrierCount, vk_memoryBarriers,
        bufferMemoryBarrierCount, vk_bufferMemoryBarriers,
        imageMemoryBarrierCount, vk_imageMemoryBarriers);

    return true;
}

bool CVulkanCommandBuffer::beginRenderPass_impl(const SRenderpassBeginInfo* pRenderPassBegin, asset::E_SUBPASS_CONTENTS content)
{
    constexpr uint32_t MAX_CLEAR_VALUE_COUNT = (1 << 12ull) / sizeof(VkClearValue);
    VkClearValue vk_clearValues[MAX_CLEAR_VALUE_COUNT];
    assert(pRenderPassBegin->clearValueCount <= MAX_CLEAR_VALUE_COUNT);

    for (uint32_t i = 0u; i < pRenderPassBegin->clearValueCount; ++i)
    {
        for (uint32_t k = 0u; k < 4u; ++k)
            vk_clearValues[i].color.uint32[k] = pRenderPassBegin->clearValues[i].color.uint32[k];

        vk_clearValues[i].depthStencil.depth = pRenderPassBegin->clearValues[i].depthStencil.depth;
        vk_clearValues[i].depthStencil.stencil = pRenderPassBegin->clearValues[i].depthStencil.stencil;
    }

    VkRenderPassBeginInfo vk_beginInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    vk_beginInfo.pNext = nullptr;
    vk_beginInfo.renderPass = IBackendObject::compatibility_cast<const CVulkanRenderpass*>(pRenderPassBegin->renderpass.get(), this)->getInternalObject();
    vk_beginInfo.framebuffer = IBackendObject::compatibility_cast<const CVulkanFramebuffer*>(pRenderPassBegin->framebuffer.get(), this)->getInternalObject();
    vk_beginInfo.renderArea = pRenderPassBegin->renderArea;
    vk_beginInfo.clearValueCount = pRenderPassBegin->clearValueCount;
    vk_beginInfo.pClearValues = vk_clearValues;

    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    vk->vk.vkCmdBeginRenderPass(m_cmdbuf, &vk_beginInfo, static_cast<VkSubpassContents>(content));

    return true;
}

bool CVulkanCommandBuffer::bindDescriptorSets_impl(asset::E_PIPELINE_BIND_POINT pipelineBindPoint,
    const pipeline_layout_t* layout, uint32_t firstSet, uint32_t descriptorSetCount,
    const descriptor_set_t* const* const pDescriptorSets,
    const uint32_t dynamicOffsetCount, const uint32_t* dynamicOffsets)
{
    VkPipelineLayout vk_pipelineLayout = IBackendObject::compatibility_cast<const CVulkanPipelineLayout*>(layout, this)->getInternalObject();

    uint32_t dynamicOffsetCountPerSet[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT] = {};

    VkDescriptorSet vk_descriptorSets[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT] = {};
    for (uint32_t i = 0u; i < descriptorSetCount; ++i)
    {
        if (pDescriptorSets[i] && pDescriptorSets[i]->getAPIType() == EAT_VULKAN)
        {
            vk_descriptorSets[i] = IBackendObject::compatibility_cast<const CVulkanDescriptorSet*>(pDescriptorSets[i], this)->getInternalObject();

            if (dynamicOffsets) // count dynamic offsets per set, if there are any
            {
                dynamicOffsetCountPerSet[i] += pDescriptorSets[i]->getLayout()->getDescriptorRedirect(asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER_DYNAMIC).getTotalCount();
                dynamicOffsetCountPerSet[i] += pDescriptorSets[i]->getLayout()->getDescriptorRedirect(asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER_DYNAMIC).getTotalCount();
            }
        }
    }

    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();

    // We allow null descriptor sets in our bind function to skip a certain set number we don't use
    // Will bind [first, last) with one call
    uint32_t dynamicOffsetsBindOffset = 0u;
    uint32_t bindCallsCount = 0u;
    uint32_t first = ~0u;
    uint32_t last = ~0u;
    for (uint32_t i = 0u; i < descriptorSetCount; ++i)
    {
        if (pDescriptorSets[i])
        {
            if (first == last)
            {
                first = i;
                last = first + 1;
            }
            else
                ++last;

            // Do a look ahead
            if ((i + 1 >= descriptorSetCount) || !pDescriptorSets[i + 1])
            {
                if (dynamicOffsets)
                {
                    uint32_t dynamicOffsetCount = 0u;
                    for (uint32_t setIndex = first; setIndex < last; ++setIndex)
                        dynamicOffsetCount += dynamicOffsetCountPerSet[setIndex];

                    vk->vk.vkCmdBindDescriptorSets(
                        m_cmdbuf,
                        static_cast<VkPipelineBindPoint>(pipelineBindPoint),
                        vk_pipelineLayout,
                        // firstSet + first, last - first, vk_descriptorSets + first, vk_dynamicOffsetCount, vk_dynamicOffsets);
                        firstSet + first, last - first, vk_descriptorSets + first,
                        dynamicOffsetCount, dynamicOffsets + dynamicOffsetsBindOffset);

                    dynamicOffsetsBindOffset += dynamicOffsetCount;
                }
                else
                {
                    vk->vk.vkCmdBindDescriptorSets(
                        m_cmdbuf,
                        static_cast<VkPipelineBindPoint>(pipelineBindPoint),
                        vk_pipelineLayout,
                        firstSet + first, last - first, vk_descriptorSets + first, 0u, nullptr);
                }

                first = ~0u;
                last = ~0u;
                ++bindCallsCount;
            }
        }
    }

    // with K slots you need at most (K+1)/2 calls
    assert(bindCallsCount <= (IGPUPipelineLayout::DESCRIPTOR_SET_COUNT + 1) / 2);

    return true;
}

bool CVulkanCommandBuffer::clearColorImage_impl(image_t* image, asset::IImage::E_LAYOUT imageLayout, const asset::SClearColorValue* pColor, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges)
{
    VkClearColorValue vk_clearColorValue;
    for (uint32_t k = 0u; k < 4u; ++k)
        vk_clearColorValue.uint32[k] = pColor->uint32[k];

    constexpr uint32_t MAX_COUNT = (1u << 12) / sizeof(VkImageSubresourceRange);
    assert(rangeCount <= MAX_COUNT);
    VkImageSubresourceRange vk_ranges[MAX_COUNT];

    for (uint32_t i = 0u; i < rangeCount; ++i)
    {
        vk_ranges[i].aspectMask = static_cast<VkImageAspectFlags>(pRanges[i].aspectMask.value);
        vk_ranges[i].baseMipLevel = pRanges[i].baseMipLevel;
        vk_ranges[i].levelCount = pRanges[i].layerCount;
        vk_ranges[i].baseArrayLayer = pRanges[i].baseArrayLayer;
        vk_ranges[i].layerCount = pRanges[i].layerCount;
    }

    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    vk->vk.vkCmdClearColorImage(
        m_cmdbuf,
        IBackendObject::compatibility_cast<const CVulkanImage*>(image, this)->getInternalObject(),
        static_cast<VkImageLayout>(imageLayout),
        &vk_clearColorValue,
        rangeCount,
        vk_ranges);

    return true;
}

bool CVulkanCommandBuffer::clearDepthStencilImage_impl(image_t* image, asset::IImage::E_LAYOUT imageLayout, const asset::SClearDepthStencilValue* pDepthStencil, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges)
{
    VkClearDepthStencilValue vk_clearDepthStencilValue = { pDepthStencil[0].depth, pDepthStencil[0].stencil };

    constexpr uint32_t MAX_COUNT = (1u << 12) / sizeof(VkImageSubresourceRange);
    assert(rangeCount <= MAX_COUNT);
    VkImageSubresourceRange vk_ranges[MAX_COUNT];

    for (uint32_t i = 0u; i < rangeCount; ++i)
    {
        vk_ranges[i].aspectMask = static_cast<VkImageAspectFlags>(pRanges[i].aspectMask.value);
        vk_ranges[i].baseMipLevel = pRanges[i].baseMipLevel;
        vk_ranges[i].levelCount = pRanges[i].layerCount;
        vk_ranges[i].baseArrayLayer = pRanges[i].baseArrayLayer;
        vk_ranges[i].layerCount = pRanges[i].layerCount;
    }

    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    vk->vk.vkCmdClearDepthStencilImage(
        m_cmdbuf,
        IBackendObject::compatibility_cast<const CVulkanImage*>(image, this)->getInternalObject(),
        static_cast<VkImageLayout>(imageLayout),
        &vk_clearDepthStencilValue,
        rangeCount,
        vk_ranges);

    return true;
}

bool CVulkanCommandBuffer::clearAttachments(uint32_t attachmentCount, const asset::SClearAttachment* pAttachments, uint32_t rectCount, const asset::SClearRect* pRects)
{
    constexpr uint32_t MAX_ATTACHMENT_COUNT = 8u;
    assert(attachmentCount <= MAX_ATTACHMENT_COUNT);
    VkClearAttachment vk_clearAttachments[MAX_ATTACHMENT_COUNT];

    constexpr uint32_t MAX_REGION_PER_ATTACHMENT_COUNT = ((1u << 12) - sizeof(vk_clearAttachments)) / sizeof(VkClearRect);
    assert(rectCount <= MAX_REGION_PER_ATTACHMENT_COUNT);
    VkClearRect vk_clearRects[MAX_REGION_PER_ATTACHMENT_COUNT];

    for (uint32_t i = 0u; i < attachmentCount; ++i)
    {
        vk_clearAttachments[i].aspectMask = static_cast<VkImageAspectFlags>(pAttachments[i].aspectMask);
        vk_clearAttachments[i].colorAttachment = pAttachments[i].colorAttachment;

        auto& vk_clearValue = vk_clearAttachments[i].clearValue;
        const auto& clearValue = pAttachments[i].clearValue;

        for (uint32_t k = 0u; k < 4u; ++k)
            vk_clearValue.color.uint32[k] = clearValue.color.uint32[k];

        vk_clearValue.depthStencil.depth = clearValue.depthStencil.depth;
        vk_clearValue.depthStencil.stencil = clearValue.depthStencil.stencil;
    }

    for (uint32_t i = 0u; i < rectCount; ++i)
    {
        vk_clearRects[i].rect = pRects[i].rect;
        vk_clearRects[i].baseArrayLayer = pRects[i].baseArrayLayer;
        vk_clearRects[i].layerCount = pRects[i].layerCount;
    }

    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    vk->vk.vkCmdClearAttachments(
        m_cmdbuf,
        attachmentCount,
        vk_clearAttachments,
        rectCount,
        vk_clearRects);

    return true;
}

bool CVulkanCommandBuffer::executeCommands_impl(uint32_t count, cmdbuf_t* const* const cmdbufs)
{
    constexpr uint32_t MAX_COMMAND_BUFFER_COUNT = (1ull << 12) / sizeof(void*);
    assert(count <= MAX_COMMAND_BUFFER_COUNT);

    VkCommandBuffer vk_commandBuffers[MAX_COMMAND_BUFFER_COUNT];

    for (uint32_t i = 0u; i < count; ++i)
        vk_commandBuffers[i] = IBackendObject::compatibility_cast<const CVulkanCommandBuffer*>(cmdbufs[i], this)->getInternalObject();

    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    vk->vk.vkCmdExecuteCommands(m_cmdbuf, count, vk_commandBuffers);

    return true;
}

static std::vector<core::smart_refctd_ptr<const core::IReferenceCounted>> getBuildGeometryInfoReferences(const IGPUAccelerationStructure::DeviceBuildGeometryInfo& info)
{   
    // TODO: Use Better Container than Vector
    std::vector<core::smart_refctd_ptr<const core::IReferenceCounted>> ret;
        
    static constexpr size_t MaxGeometryPerBuildInfoCount = 64;
    // + 3 because of info.srcAS + info.dstAS + info.scratchAddr.buffer
    // * 3 because of worst-case all triangle data ( vertexData + indexData + transformData+
    ret.resize(MaxGeometryPerBuildInfoCount * 3 + 3); 

    ret.push_back(core::smart_refctd_ptr<const IGPUAccelerationStructure>(info.srcAS));
    ret.push_back(core::smart_refctd_ptr<const IGPUAccelerationStructure>(info.dstAS));
    ret.push_back(info.scratchAddr.buffer);
                
    if(!info.geometries.empty())
    {
        IGPUAccelerationStructure::Geometry<IGPUAccelerationStructure::DeviceAddressType>* geoms = info.geometries.begin();
        for(uint32_t g = 0; g < info.geometries.size(); ++g)
        {
            auto const & geometry = geoms[g];
            if(IGPUAccelerationStructure::E_GEOM_TYPE::EGT_TRIANGLES == geometry.type)
            {
                auto const & triangles = geometry.data.triangles;
                if (triangles.vertexData.isValid())
                    ret.push_back(triangles.vertexData.buffer);
                if (triangles.indexData.isValid())
                    ret.push_back(triangles.indexData.buffer);
                if (triangles.transformData.isValid())
                    ret.push_back(triangles.transformData.buffer);
            }
            else if(IGPUAccelerationStructure::E_GEOM_TYPE::EGT_AABBS == geometry.type)
            {
                const auto & aabbs = geometry.data.aabbs;
                if (aabbs.data.isValid())
                    ret.push_back(aabbs.data.buffer);
            }
            else if(IGPUAccelerationStructure::E_GEOM_TYPE::EGT_INSTANCES == geometry.type)
            {
                const auto & instances = geometry.data.instances;
                if (instances.data.isValid())
                    ret.push_back(instances.data.buffer);
            }
        }
    }
    return ret;
}

bool CVulkanCommandBuffer::buildAccelerationStructures_impl(const core::SRange<IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, IGPUAccelerationStructure::BuildRangeInfo* const* ppBuildRangeInfos)
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    VkDevice vk_device = vulkanDevice->getInternalObject();
    auto* vk = vulkanDevice->getFunctionTable();

    static constexpr size_t MaxGeometryPerBuildInfoCount = 64;
    static constexpr size_t MaxBuildInfoCount = 128;
    size_t infoCount = pInfos.size();
    assert(infoCount <= MaxBuildInfoCount);

    // TODO: Use better container when ready for these stack allocated memories.
    VkAccelerationStructureBuildGeometryInfoKHR vk_buildGeomsInfos[MaxBuildInfoCount] = {};

    uint32_t geometryArrayOffset = 0u;
    VkAccelerationStructureGeometryKHR vk_geometries[MaxGeometryPerBuildInfoCount * MaxBuildInfoCount] = {};

    IGPUAccelerationStructure::DeviceBuildGeometryInfo* infos = pInfos.begin();

    for(uint32_t i = 0; i < infoCount; ++i)
    {
        uint32_t geomCount = infos[i].geometries.size();
        vk_buildGeomsInfos[i] = CVulkanAccelerationStructure::getVkASBuildGeomInfoFromBuildGeomInfo(vk_device, vk, infos[i], &vk_geometries[geometryArrayOffset]);
        geometryArrayOffset += geomCount;
    }

    static_assert(sizeof(IGPUAccelerationStructure::BuildRangeInfo) == sizeof(VkAccelerationStructureBuildRangeInfoKHR));
    auto buildRangeInfos = reinterpret_cast<const VkAccelerationStructureBuildRangeInfoKHR* const*>(ppBuildRangeInfos);
    vk->vk.vkCmdBuildAccelerationStructuresKHR(m_cmdbuf, infoCount, vk_buildGeomsInfos, buildRangeInfos);
    
    return true;
}
    
bool CVulkanCommandBuffer::buildAccelerationStructuresIndirect_impl(
    const core::SRange<IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, 
    const core::SRange<IGPUAccelerationStructure::DeviceAddressType>& pIndirectDeviceAddresses,
    const uint32_t* pIndirectStrides,
    const uint32_t* const* ppMaxPrimitiveCounts)
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    VkDevice vk_device = vulkanDevice->getInternalObject();
    auto* vk = vulkanDevice->getFunctionTable();

    static constexpr size_t MaxGeometryPerBuildInfoCount = 64;
    static constexpr size_t MaxBuildInfoCount = 128;
    size_t infoCount = pInfos.size();
    size_t indirectDeviceAddressesCount = pIndirectDeviceAddresses.size();
    assert(infoCount <= MaxBuildInfoCount);
    assert(infoCount == indirectDeviceAddressesCount);
                
    // TODO: Use better container when ready for these stack allocated memories.
    VkAccelerationStructureBuildGeometryInfoKHR vk_buildGeomsInfos[MaxBuildInfoCount] = {};
    VkDeviceSize vk_indirectDeviceAddresses[MaxBuildInfoCount] = {};

    uint32_t geometryArrayOffset = 0u;
    VkAccelerationStructureGeometryKHR vk_geometries[MaxGeometryPerBuildInfoCount * MaxBuildInfoCount] = {};

    IGPUAccelerationStructure::DeviceBuildGeometryInfo* infos = pInfos.begin();
    IGPUAccelerationStructure::DeviceAddressType* indirectDeviceAddresses = pIndirectDeviceAddresses.begin();
    for(uint32_t i = 0; i < infoCount; ++i)
    {
        uint32_t geomCount = infos[i].geometries.size();

        vk_buildGeomsInfos[i] = CVulkanAccelerationStructure::getVkASBuildGeomInfoFromBuildGeomInfo(vk_device, vk, infos[i], &vk_geometries[geometryArrayOffset]);
        geometryArrayOffset += geomCount;

        auto addr = CVulkanAccelerationStructure::getVkDeviceOrHostAddress<IGPUAccelerationStructure::DeviceAddressType>(vk_device, vk, indirectDeviceAddresses[i]);
        vk_indirectDeviceAddresses[i] = addr.deviceAddress;
    }
                
    vk->vk.vkCmdBuildAccelerationStructuresIndirectKHR(m_cmdbuf, infoCount, vk_buildGeomsInfos, vk_indirectDeviceAddresses, pIndirectStrides, ppMaxPrimitiveCounts);
    return true;
}

bool CVulkanCommandBuffer::copyAccelerationStructure_impl(const IGPUAccelerationStructure::CopyInfo& copyInfo)
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    VkDevice vk_device = vulkanDevice->getInternalObject();
    auto* vk = vulkanDevice->getFunctionTable();

    VkCopyAccelerationStructureInfoKHR info = CVulkanAccelerationStructure::getVkASCopyInfo(vk_device, vk, copyInfo);
    vk->vk.vkCmdCopyAccelerationStructureKHR(m_cmdbuf, &info);
    return true;
}
    
bool CVulkanCommandBuffer::copyAccelerationStructureToMemory_impl(const IGPUAccelerationStructure::DeviceCopyToMemoryInfo& copyInfo)
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    VkDevice vk_device = vulkanDevice->getInternalObject();
    auto* vk = vulkanDevice->getFunctionTable();
            
    VkCopyAccelerationStructureToMemoryInfoKHR info = CVulkanAccelerationStructure::getVkASCopyToMemoryInfo(vk_device, vk, copyInfo);
    vk->vk.vkCmdCopyAccelerationStructureToMemoryKHR(m_cmdbuf, &info);
    return true;
}

bool CVulkanCommandBuffer::copyAccelerationStructureFromMemory_impl(const IGPUAccelerationStructure::DeviceCopyFromMemoryInfo& copyInfo)
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    VkDevice vk_device = vulkanDevice->getInternalObject();
    auto* vk = vulkanDevice->getFunctionTable();
            
    VkCopyMemoryToAccelerationStructureInfoKHR info = CVulkanAccelerationStructure::getVkASCopyFromMemoryInfo(vk_device, vk, copyInfo);
    vk->vk.vkCmdCopyMemoryToAccelerationStructureKHR(m_cmdbuf, &info);
    return true;
}
    
bool CVulkanCommandBuffer::resetQueryPool_impl(IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount)
{
    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    auto vk_queryPool = IBackendObject::compatibility_cast<CVulkanQueryPool*>(queryPool, this)->getInternalObject();
    vk->vk.vkCmdResetQueryPool(m_cmdbuf, vk_queryPool, firstQuery, queryCount);

    return true;
}

bool CVulkanCommandBuffer::beginQuery_impl(IQueryPool* queryPool, uint32_t query, core::bitflag<IQueryPool::E_QUERY_CONTROL_FLAGS> flags)
{
    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    auto vk_queryPool = IBackendObject::compatibility_cast<CVulkanQueryPool*>(queryPool, this)->getInternalObject();
    auto vk_flags = CVulkanQueryPool::getVkQueryControlFlagsFromQueryControlFlags(flags.value);
    vk->vk.vkCmdBeginQuery(m_cmdbuf, vk_queryPool, query, vk_flags);

    return true;
}

bool CVulkanCommandBuffer::endQuery_impl(IQueryPool* queryPool, uint32_t query)
{
    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    auto vk_queryPool = IBackendObject::compatibility_cast<CVulkanQueryPool*>(queryPool, this)->getInternalObject();
    vk->vk.vkCmdEndQuery(m_cmdbuf, vk_queryPool, query);

    return true;
}

bool CVulkanCommandBuffer::copyQueryPoolResults_impl(IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, buffer_t* dstBuffer, size_t dstOffset, size_t stride, core::bitflag<IQueryPool::E_QUERY_RESULTS_FLAGS> flags)
{
    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    auto vk_queryPool = IBackendObject::compatibility_cast<CVulkanQueryPool*>(queryPool, this)->getInternalObject();
    auto vk_dstBuffer = IBackendObject::compatibility_cast<CVulkanBuffer*>(dstBuffer, this)->getInternalObject();
    auto vk_queryResultsFlags = CVulkanQueryPool::getVkQueryResultsFlagsFromQueryResultsFlags(flags.value); 
    vk->vk.vkCmdCopyQueryPoolResults(m_cmdbuf, vk_queryPool, firstQuery, queryCount, vk_dstBuffer, dstOffset, static_cast<VkDeviceSize>(stride), vk_queryResultsFlags);
        
    return true;
}

bool CVulkanCommandBuffer::writeTimestamp_impl(asset::E_PIPELINE_STAGE_FLAGS pipelineStage, IQueryPool* queryPool, uint32_t query)
{
    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
    auto vk_queryPool = IBackendObject::compatibility_cast<CVulkanQueryPool*>(queryPool, this)->getInternalObject();
    auto vk_pipelineStageFlagBit = static_cast<VkPipelineStageFlagBits>(getVkPipelineStageFlagsFromPipelineStageFlags(pipelineStage));
    vk->vk.vkCmdWriteTimestamp(m_cmdbuf, vk_pipelineStageFlagBit, vk_queryPool, query);

    return true;
}

bool CVulkanCommandBuffer::writeAccelerationStructureProperties_impl(const core::SRange<IGPUAccelerationStructure>& pAccelerationStructures, IQueryPool::E_QUERY_TYPE queryType, IQueryPool* queryPool, uint32_t firstQuery) 
{
    // TODO: Use Better Containers
    static constexpr size_t MaxAccelerationStructureCount = 128;
    uint32_t asCount = static_cast<uint32_t>(pAccelerationStructures.size());
    assert(asCount <= MaxAccelerationStructureCount);
    auto accelerationStructures = pAccelerationStructures.begin();

    VkAccelerationStructureKHR vk_accelerationStructures[MaxAccelerationStructureCount] = {};
    for(size_t i = 0; i < asCount; ++i) 
        vk_accelerationStructures[i] = IBackendObject::compatibility_cast<CVulkanAccelerationStructure*>(&accelerationStructures[i], this)->getInternalObject();
            
    const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();

    auto vk_queryPool = IBackendObject::compatibility_cast<CVulkanQueryPool*>(queryPool, this)->getInternalObject();
    auto vk_queryType = CVulkanQueryPool::getVkQueryTypeFromQueryType(queryType);
    vk->vk.vkCmdWriteAccelerationStructuresPropertiesKHR(m_cmdbuf, asCount, vk_accelerationStructures, vk_queryType, vk_queryPool, firstQuery);

    return true;
}

}