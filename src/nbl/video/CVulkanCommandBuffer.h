#ifndef __NBL_C_VULKAN_COMMAND_BUFFER_H_INCLUDED__
#define __NBL_C_VULKAN_COMMAND_BUFFER_H_INCLUDED__

#include "nbl/video/IGPUCommandBuffer.h"

#include "nbl/video/CVulkanBuffer.h"
#include "nbl/video/CVulkanImage.h"
#include "nbl/video/CVulkanComputePipeline.h"
#include "nbl/video/CVulkanPipelineLayout.h"
#include "nbl/video/CVulkanDescriptorSet.h"
#include "nbl/video/CVulkanFramebuffer.h"
#include "nbl/video/CVulkanRenderpass.h"
#include "nbl/video/CVulkanLogicalDevice.h"
#include "nbl/video/CVulkanEvent.h"

#include <volk.h>

namespace nbl::video
{
struct ArgumentReferenceSegment;

class CVulkanCommandBuffer : public IGPUCommandBuffer
{
public:
    CVulkanCommandBuffer(core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, E_LEVEL level,
        VkCommandBuffer _vkcmdbuf, core::smart_refctd_ptr<IGPUCommandPool>&& commandPool, system::logger_opt_smart_ptr&& logger)
        : IGPUCommandBuffer(std::move(logicalDevice), level, std::move(commandPool), std::move(logger)), m_cmdbuf(_vkcmdbuf)
    {
        if (m_cmdpool->getAPIType() == EAT_VULKAN)
        {
            CVulkanCommandPool* vulkanCommandPool = static_cast<CVulkanCommandPool*>(m_cmdpool.get());
            vulkanCommandPool->emplace_n(m_argListTail, nullptr, nullptr);
            m_argListHead = m_argListTail;
        }
    }

    ~CVulkanCommandBuffer()
    {
        freeSpaceInCmdPool();
    }

    bool begin_impl(core::bitflag<E_USAGE> recordingFlags, const SInheritanceInfo* inheritanceInfo) override
    {
        VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.pNext = nullptr; // pNext must be NULL or a pointer to a valid instance of VkDeviceGroupCommandBufferBeginInfo
        beginInfo.flags = static_cast<VkCommandBufferUsageFlags>(recordingFlags.value);

        VkCommandBufferInheritanceInfo vk_inheritanceInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO };
        if (inheritanceInfo)
        {
            // TODO(achal): Remove
            core::smart_refctd_ptr<const core::IReferenceCounted> tmp[2] = { inheritanceInfo->renderpass, inheritanceInfo->framebuffer };

            vk_inheritanceInfo.pNext = nullptr;
            if (!inheritanceInfo->renderpass || inheritanceInfo->renderpass->getAPIType() != EAT_VULKAN || !inheritanceInfo->renderpass->isCompatibleDevicewise(this))
                return false;

            // if (!inheritanceInfo->framebuffer || inheritanceInfo->framebuffer->getAPIType() != EAT_VULKAN || !inheritanceInfo->framebuffer->isCompatibleDevicewise(this))
            //     return false;

            // TODO(achal): Remove
            // if (!saveReferencesToResources(tmp, tmp + 2))
            if (!saveReferencesToResources(tmp, tmp + 1))
                return false;

            vk_inheritanceInfo.renderPass = IBackendObject::compatibility_cast<const CVulkanRenderpass*>(inheritanceInfo->renderpass.get(), this)->getInternalObject();
            vk_inheritanceInfo.subpass = inheritanceInfo->subpass;
            // Todo(achal):
            // From the spec:
            // Specifying the exact framebuffer that the secondary command buffer will be
            // executed with may result in better performance at command buffer execution time.
            vk_inheritanceInfo.framebuffer = VK_NULL_HANDLE; // IBackendObject::compatibility_cast<const CVulkanFramebuffer*>(inheritanceInfo->framebuffer.get(), this)->getInternalObject();
            vk_inheritanceInfo.occlusionQueryEnable = inheritanceInfo->occlusionQueryEnable;
            vk_inheritanceInfo.queryFlags = static_cast<VkQueryControlFlags>(inheritanceInfo->queryFlags.value);
            vk_inheritanceInfo.pipelineStatistics = static_cast<VkQueryPipelineStatisticFlags>(0u); // must be 0
        }
        beginInfo.pInheritanceInfo = inheritanceInfo ? &vk_inheritanceInfo : nullptr;
        
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        const VkResult retval = vk->vk.vkBeginCommandBuffer(m_cmdbuf, &beginInfo);
        return retval == VK_SUCCESS;
    }

    bool end_impl() override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        VkResult retval = vk->vk.vkEndCommandBuffer(m_cmdbuf);
        return retval == VK_SUCCESS;
    }

    bool reset_impl(core::bitflag<E_RESET_FLAGS> flags) override
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        const VkResult result = vk->vk.vkResetCommandBuffer(m_cmdbuf, static_cast<VkCommandBufferResetFlags>(flags.value));
        return result == VK_SUCCESS;
    }

    // TODO(achal): This entire function is temporary. Vulkan doesn't need to do anything after IGPUCommandBuffer::releaseResouurcesBackToPool.
    // I will remove before merge.
    void releaseResourcesBackToPool_impl() override
    {
        // TODO(achal): This call is temporary. It frees the old Vulkan-specific segmented list which is still around 
        // just for testing. I will remove this before the merge.
        freeSpaceInCmdPool();
    }

    void bindIndexBuffer_impl(const buffer_t* buffer, size_t offset, asset::E_INDEX_TYPE indexType) override
    {
        assert(indexType < asset::EIT_UNKNOWN);

        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();

        vk->vk.vkCmdBindIndexBuffer(
            m_cmdbuf,
            IBackendObject::compatibility_cast<const CVulkanBuffer*>(buffer, this)->getInternalObject(),
            static_cast<VkDeviceSize>(offset),
            static_cast<VkIndexType>(indexType));
    }

    bool draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) override
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdDraw(m_cmdbuf, vertexCount, instanceCount, firstVertex, firstInstance);
        return true;
    }

    bool drawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) override
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdDrawIndexed(m_cmdbuf, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
        return true;
    }

    void drawIndirect_impl(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride) override
    {
        const core::smart_refctd_ptr<const core::IReferenceCounted> tmp[1] = {
            core::smart_refctd_ptr<const IGPUBuffer>(buffer) };

        // TODO(achal): Remove.
        saveReferencesToResources(tmp, tmp + 1);

        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdDrawIndirect(
            m_cmdbuf,
            IBackendObject::compatibility_cast<const CVulkanBuffer*>(buffer, this)->getInternalObject(),
            static_cast<VkDeviceSize>(offset),
            drawCount,
            stride);
    }

    void drawIndexedIndirect_impl(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride) override
    {
        const core::smart_refctd_ptr<const core::IReferenceCounted> tmp[1] = {
            core::smart_refctd_ptr<const IGPUBuffer>(buffer) };

        // TODO(achal): Remove.
        saveReferencesToResources(tmp, tmp + 1);

        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdDrawIndexedIndirect(
            m_cmdbuf,
            IBackendObject::compatibility_cast<const CVulkanBuffer*>(buffer, this)->getInternalObject(),
            static_cast<VkDeviceSize>(offset),
            drawCount,
            stride);
    }

    void drawIndirectCount_impl(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride) override
    {
        const core::smart_refctd_ptr<const core::IReferenceCounted> tmp[2] = {
            core::smart_refctd_ptr<const IGPUBuffer>(buffer),
            core::smart_refctd_ptr<const IGPUBuffer>(countBuffer) };

        // TODO(achal): Remove.
        saveReferencesToResources(tmp, tmp + 2);

        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdDrawIndirectCount(
            m_cmdbuf,
            IBackendObject::compatibility_cast<const CVulkanBuffer*>(buffer, this)->getInternalObject(),
            static_cast<VkDeviceSize>(offset),
            IBackendObject::compatibility_cast<const CVulkanBuffer*>(countBuffer, this)->getInternalObject(),
            static_cast<VkDeviceSize>(countBufferOffset),
            maxDrawCount,
            stride);
    }

    void drawIndexedIndirectCount_impl(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride) override
    {
        const core::smart_refctd_ptr<const core::IReferenceCounted> tmp[2] = {
            core::smart_refctd_ptr<const IGPUBuffer>(buffer),
            core::smart_refctd_ptr<const IGPUBuffer>(countBuffer) };

        // TODO(achal): Remove.
        saveReferencesToResources(tmp, tmp + 2);

        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdDrawIndexedIndirectCount(
            m_cmdbuf,
            IBackendObject::compatibility_cast<const CVulkanBuffer*>(buffer, this)->getInternalObject(),
            static_cast<VkDeviceSize>(offset),
            IBackendObject::compatibility_cast<const CVulkanBuffer*>(countBuffer, this)->getInternalObject(),
            static_cast<VkDeviceSize>(countBufferOffset),
            maxDrawCount,
            stride);
    }

    bool drawMeshBuffer(const IGPUMeshBuffer::base_t* meshBuffer) override
    {
        if (!meshBuffer || !meshBuffer->getInstanceCount())
            return false;

        const auto* pipeline = meshBuffer->getPipeline();
        const auto bindingFlags = pipeline->getVertexInputParams().enabledBindingFlags;
        auto vertexBufferBindings = meshBuffer->getVertexBufferBindings();
        auto indexBufferBinding = meshBuffer->getIndexBufferBinding();
        const auto indexType = meshBuffer->getIndexType();

        const video::IGPUBuffer* gpuBufferBindings[asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT];
        {
            for (size_t i = 0; i < nbl::asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; ++i)
                gpuBufferBindings[i] = vertexBufferBindings[i].buffer.get();
        }

        size_t bufferBindingsOffsets[asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT];
        {
            for (size_t i = 0; i < asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; ++i)
                bufferBindingsOffsets[i] = vertexBufferBindings[i].offset;
        }

        bindVertexBuffers(0, asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT, gpuBufferBindings, bufferBindingsOffsets);
        bindIndexBuffer(indexBufferBinding.buffer.get(), indexBufferBinding.offset, indexType);

        const bool isIndexed = indexType != asset::EIT_UNKNOWN;

        const size_t instanceCount = meshBuffer->getInstanceCount();
        const size_t firstInstance = meshBuffer->getBaseInstance();
        const size_t firstVertex = meshBuffer->getBaseVertex();

        if (isIndexed)
        {
            const size_t& indexCount = meshBuffer->getIndexCount();
            const size_t firstIndex = 0; // I don't think we have utility telling us this one
            const size_t& vertexOffset = firstVertex;

            return drawIndexed(indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
        }
        else
        {
            const size_t& vertexCount = meshBuffer->getIndexCount();

            return draw(vertexCount, instanceCount, firstVertex, firstInstance);
        }
    }

    bool setViewport(uint32_t firstViewport, uint32_t viewportCount, const asset::SViewport* pViewports) override
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

    bool setLineWidth(float lineWidth) override
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdSetLineWidth(m_cmdbuf, lineWidth);
        return true;
    }

    bool setDepthBias(float depthBiasConstantFactor, float depthBiasClamp, float depthBiasSlopeFactor) override
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdSetDepthBias(m_cmdbuf, depthBiasConstantFactor, depthBiasClamp, depthBiasSlopeFactor);
        return true;
    }

    bool setBlendConstants(const float blendConstants[4]) override
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdSetBlendConstants(m_cmdbuf, blendConstants);
        return true;
    }

    bool copyBuffer(const buffer_t* srcBuffer, buffer_t* dstBuffer, uint32_t regionCount, const asset::SBufferCopy* pRegions) override
    {
        if (!srcBuffer || srcBuffer->getAPIType() != EAT_VULKAN)
            return false;

        if (!dstBuffer || dstBuffer->getAPIType() != EAT_VULKAN)
            return false;

        const core::smart_refctd_ptr<const core::IReferenceCounted> tmp[2] = {
            core::smart_refctd_ptr<const IGPUBuffer>(srcBuffer),
            core::smart_refctd_ptr<const IGPUBuffer>(dstBuffer) };

        if (!saveReferencesToResources(tmp, tmp + 2))
            return false;

        VkBuffer vk_srcBuffer = IBackendObject::compatibility_cast<const CVulkanBuffer*>(srcBuffer, this)->getInternalObject();
        VkBuffer vk_dstBuffer = IBackendObject::compatibility_cast<const CVulkanBuffer*>(dstBuffer, this)->getInternalObject();

        constexpr uint32_t MAX_BUFFER_COPY_REGION_COUNT = 681u;
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

    bool copyImage(const image_t* srcImage, asset::E_IMAGE_LAYOUT srcImageLayout, image_t* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SImageCopy* pRegions) override
    {
        if (!srcImage || srcImage->getAPIType() != EAT_VULKAN)
            return false;

        if (!dstImage || dstImage->getAPIType() != EAT_VULKAN)
            return false;

        core::smart_refctd_ptr<const core::IReferenceCounted> tmp[2] = {
            core::smart_refctd_ptr<const IGPUImage>(srcImage),
            core::smart_refctd_ptr<const IGPUImage>(dstImage) };

        if (!saveReferencesToResources(tmp, tmp + 2))
            return false;

        constexpr uint32_t MAX_COUNT = (1u << 12) / sizeof(VkImageCopy);
        assert(regionCount <= MAX_COUNT);

        VkImageCopy vk_regions[MAX_COUNT];
        for (uint32_t i = 0u; i < regionCount; ++i)
        {
            vk_regions[i].srcSubresource.aspectMask = static_cast<VkImageAspectFlags>(pRegions[i].srcSubresource.aspectMask);
            vk_regions[i].srcSubresource.baseArrayLayer = pRegions[i].srcSubresource.baseArrayLayer;
            vk_regions[i].srcSubresource.layerCount = pRegions[i].srcSubresource.layerCount;
            vk_regions[i].srcSubresource.mipLevel = pRegions[i].srcSubresource.mipLevel;

            vk_regions[i].srcOffset = { static_cast<int32_t>(pRegions[i].srcOffset.x), static_cast<int32_t>(pRegions[i].srcOffset.y), static_cast<int32_t>(pRegions[i].srcOffset.z) };

            vk_regions[i].dstSubresource.aspectMask = static_cast<VkImageAspectFlags>(pRegions[i].dstSubresource.aspectMask);
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

    bool copyBufferToImage(const buffer_t* srcBuffer, image_t* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) override
    {
        if ((srcBuffer->getAPIType() != EAT_VULKAN) || (dstImage->getAPIType() != EAT_VULKAN))
            return false;

        const core::smart_refctd_ptr<const core::IReferenceCounted> tmp[2] =
        {
            core::smart_refctd_ptr<const video::IGPUBuffer>(srcBuffer),
            core::smart_refctd_ptr<const video::IGPUImage>(dstImage)
        };

        if (!saveReferencesToResources(tmp, tmp + 2))
            return false;

        constexpr uint32_t MAX_REGION_COUNT = (1ull << 12) / sizeof(VkBufferImageCopy);
        assert(regionCount <= MAX_REGION_COUNT);

        VkBufferImageCopy vk_regions[MAX_REGION_COUNT];
        for (uint32_t i = 0u; i < regionCount; ++i)
        {
            vk_regions[i].bufferOffset = pRegions[i].bufferOffset;
            vk_regions[i].bufferRowLength = pRegions[i].bufferRowLength;
            vk_regions[i].bufferImageHeight = pRegions[i].bufferImageHeight;
            vk_regions[i].imageSubresource.aspectMask = static_cast<VkImageAspectFlags>(pRegions[i].imageSubresource.aspectMask);
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

    bool copyImageToBuffer(const image_t* srcImage, asset::E_IMAGE_LAYOUT srcImageLayout, buffer_t* dstBuffer, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) override
    {
        if (!srcImage || (srcImage->getAPIType() != EAT_VULKAN))
            return false;

        if (!dstBuffer || (dstBuffer->getAPIType() != EAT_VULKAN))
            return false;

        core::smart_refctd_ptr<const core::IReferenceCounted> tmp[2] =
        {
            core::smart_refctd_ptr<const image_t>(srcImage),
            core::smart_refctd_ptr<const buffer_t>(dstBuffer)
        };

        if (!saveReferencesToResources(tmp, tmp + 2))
            return false;

        VkImage vk_srcImage = IBackendObject::compatibility_cast<const CVulkanImage*>(srcImage, this)->getInternalObject();
        VkBuffer vk_dstBuffer = IBackendObject::compatibility_cast<const CVulkanBuffer*>(dstBuffer, this)->getInternalObject();

        constexpr uint32_t MAX_REGION_COUNT = (1u << 12)/sizeof(VkBufferImageCopy);
        VkBufferImageCopy vk_copyRegions[MAX_REGION_COUNT];
        assert(regionCount <= MAX_REGION_COUNT);

        for (uint32_t i = 0u; i < regionCount; ++i)
        {
            vk_copyRegions[i].bufferOffset = static_cast<VkDeviceSize>(pRegions[i].bufferOffset);
            vk_copyRegions[i].bufferRowLength = pRegions[i].bufferRowLength;
            vk_copyRegions[i].bufferImageHeight = pRegions[i].bufferImageHeight;
            vk_copyRegions[i].imageSubresource.aspectMask = static_cast<VkImageAspectFlags>(pRegions[i].imageSubresource.aspectMask);
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

    bool blitImage(const image_t* srcImage, asset::E_IMAGE_LAYOUT srcImageLayout, image_t* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageBlit* pRegions, asset::ISampler::E_TEXTURE_FILTER filter) override
    {
        if (srcImage->getAPIType() != EAT_VULKAN || (dstImage->getAPIType() != EAT_VULKAN))
            return false;

        core::smart_refctd_ptr<const core::IReferenceCounted> tmp[2] = {
            core::smart_refctd_ptr<const IGPUImage>(srcImage),
            core::smart_refctd_ptr<const IGPUImage>(dstImage) };

        if (!saveReferencesToResources(tmp, tmp + 2))
            return false;

        VkImage vk_srcImage = IBackendObject::compatibility_cast<const CVulkanImage*>(srcImage, this)->getInternalObject();
        VkImage vk_dstImage = IBackendObject::compatibility_cast<const CVulkanImage*>(dstImage, this)->getInternalObject();

        constexpr uint32_t MAX_BLIT_REGION_COUNT = 100u;
        VkImageBlit vk_blitRegions[MAX_BLIT_REGION_COUNT];
        assert(regionCount <= MAX_BLIT_REGION_COUNT);

        for (uint32_t i = 0u; i < regionCount; ++i)
        {
            vk_blitRegions[i].srcSubresource.aspectMask = static_cast<VkImageAspectFlags>(pRegions[i].srcSubresource.aspectMask);
            vk_blitRegions[i].srcSubresource.mipLevel = pRegions[i].srcSubresource.mipLevel;
            vk_blitRegions[i].srcSubresource.baseArrayLayer = pRegions[i].srcSubresource.baseArrayLayer;
            vk_blitRegions[i].srcSubresource.layerCount = pRegions[i].srcSubresource.layerCount;

            // Todo(achal): Remove `static_cast`s
            vk_blitRegions[i].srcOffsets[0] = { static_cast<int32_t>(pRegions[i].srcOffsets[0].x), static_cast<int32_t>(pRegions[i].srcOffsets[0].y), static_cast<int32_t>(pRegions[i].srcOffsets[0].z) };
            vk_blitRegions[i].srcOffsets[1] = { static_cast<int32_t>(pRegions[i].srcOffsets[1].x), static_cast<int32_t>(pRegions[i].srcOffsets[1].y), static_cast<int32_t>(pRegions[i].srcOffsets[1].z) };

            vk_blitRegions[i].dstSubresource.aspectMask = static_cast<VkImageAspectFlags>(pRegions[i].dstSubresource.aspectMask);
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

    bool resolveImage(const image_t* srcImage, asset::E_IMAGE_LAYOUT srcImageLayout, image_t* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageResolve* pRegions) override
    {
        if (!srcImage || srcImage->getAPIType() != EAT_VULKAN)
            return false;

        if (!dstImage || dstImage->getAPIType() != EAT_VULKAN)
            return false;

        core::smart_refctd_ptr<const core::IReferenceCounted> tmp[2] = {
            core::smart_refctd_ptr<const IGPUImage>(srcImage),
            core::smart_refctd_ptr<const IGPUImage>(dstImage) };

        if (!saveReferencesToResources(tmp, tmp + 2))
            return false;

        constexpr uint32_t MAX_COUNT = (1u << 12) / sizeof(VkImageResolve);
        assert(regionCount <= MAX_COUNT);

        VkImageResolve vk_regions[MAX_COUNT];
        for (uint32_t i = 0u; i < regionCount; ++i)
        {
            vk_regions[i].srcSubresource.aspectMask = static_cast<VkImageAspectFlags>(pRegions[i].srcSubresource.aspectMask);
            vk_regions[i].srcSubresource.baseArrayLayer = pRegions[i].srcSubresource.baseArrayLayer;
            vk_regions[i].srcSubresource.layerCount = pRegions[i].srcSubresource.layerCount;
            vk_regions[i].srcSubresource.mipLevel = pRegions[i].srcSubresource.mipLevel;

            vk_regions[i].srcOffset = { static_cast<int32_t>(pRegions[i].srcOffset.x), static_cast<int32_t>(pRegions[i].srcOffset.y), static_cast<int32_t>(pRegions[i].srcOffset.z) };

            vk_regions[i].dstSubresource.aspectMask = static_cast<VkImageAspectFlags>(pRegions[i].dstSubresource.aspectMask);
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

    bool bindVertexBuffers(uint32_t firstBinding, uint32_t bindingCount, const buffer_t* const *const pBuffers, const size_t* pOffsets) override
    {
        constexpr uint32_t MAX_BUFFER_COUNT = 16u;
        assert(bindingCount <= MAX_BUFFER_COUNT);

        VkBuffer vk_buffers[MAX_BUFFER_COUNT];
        VkDeviceSize vk_offsets[MAX_BUFFER_COUNT];
        core::smart_refctd_ptr<const core::IReferenceCounted> tmp[MAX_BUFFER_COUNT];

        uint32_t actualBindingCount = 0u;
        VkBuffer dummyBuffer = VK_NULL_HANDLE;
        for (uint32_t i = 0u; i < bindingCount; ++i)
        {
            if (!pBuffers[i] || (pBuffers[i]->getAPIType() != EAT_VULKAN))
            {
                // continue;
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

                tmp[actualBindingCount] = core::smart_refctd_ptr<const IGPUBuffer>(pBuffers[i]);
                ++actualBindingCount;
            }
        }
        for (uint32_t i = 0u; i < bindingCount; ++i)
        {
            if (vk_buffers[i] == VK_NULL_HANDLE)
                vk_buffers[i] = dummyBuffer;
        }

        if (!saveReferencesToResources(tmp, tmp + actualBindingCount))
            return false;

        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdBindVertexBuffers(
            m_cmdbuf,
            firstBinding,
            bindingCount,
            vk_buffers,
            vk_offsets);
        return true;
    }

    bool setScissor(uint32_t firstScissor, uint32_t scissorCount, const VkRect2D* pScissors) override
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdSetScissor(m_cmdbuf, firstScissor, scissorCount, pScissors);
        return true;
    }

    bool setDepthBounds(float minDepthBounds, float maxDepthBounds) override
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdSetDepthBounds(m_cmdbuf, minDepthBounds, maxDepthBounds);
        return true;
    }

    bool setStencilCompareMask(asset::E_STENCIL_FACE_FLAGS faceMask, uint32_t compareMask) override
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdSetStencilCompareMask(m_cmdbuf, static_cast<VkStencilFaceFlags>(faceMask), compareMask);
        return true;
    }

    bool setStencilWriteMask(asset::E_STENCIL_FACE_FLAGS faceMask, uint32_t writeMask) override
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdSetStencilWriteMask(m_cmdbuf, static_cast<VkStencilFaceFlags>(faceMask), writeMask);
        return true;
    }

    bool setStencilReference(asset::E_STENCIL_FACE_FLAGS faceMask, uint32_t reference) override
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdSetStencilReference(m_cmdbuf, static_cast<VkStencilFaceFlags>(faceMask), reference);
        return true;
    }

    // Doesn't really require the return value here
    bool dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) override
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdDispatch(m_cmdbuf, groupCountX, groupCountY, groupCountZ);
        return true;
    }

    bool dispatchIndirect(const buffer_t* buffer, size_t offset) override
    {
        if (!buffer || buffer->getAPIType() != EAT_VULKAN)
            return false;

        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdDispatchIndirect(
            m_cmdbuf,
            IBackendObject::compatibility_cast<const CVulkanBuffer*>(buffer, this)->getInternalObject(),
            static_cast<VkDeviceSize>(offset));

        return true;
    }

    bool dispatchBase(uint32_t baseGroupX, uint32_t baseGroupY, uint32_t baseGroupZ, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) override
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdDispatchBase(m_cmdbuf, baseGroupX, baseGroupY, baseGroupZ, groupCountX, groupCountY, groupCountZ);
        return true;
    }

    bool setEvent(event_t* event, const SDependencyInfo& depInfo) override
    {
        if (!event || event->getAPIType() != EAT_VULKAN)
            return false;

        core::smart_refctd_ptr<const core::IReferenceCounted> tmp[] = { core::smart_refctd_ptr<const core::IReferenceCounted>(event) };
        if (!saveReferencesToResources(tmp, tmp + 1))
            return false;
        
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdSetEvent(
            m_cmdbuf,
            IBackendObject::compatibility_cast<const CVulkanEvent*>(event, this)->getInternalObject(),
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT); // No way to get this! SDependencyInfo is unused

        return true;
    }

    bool resetEvent(event_t* event, asset::E_PIPELINE_STAGE_FLAGS stageMask) override
    {
        if (!event || event->getAPIType() != EAT_VULKAN)
            return false;

        core::smart_refctd_ptr<const core::IReferenceCounted> tmp[] = { core::smart_refctd_ptr<const core::IReferenceCounted>(event) };
        if (!saveReferencesToResources(tmp, tmp + 1))
            return false;

        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdResetEvent(
            m_cmdbuf,
            IBackendObject::compatibility_cast<const CVulkanEvent*>(event, this)->getInternalObject(),
            getVkPipelineStageFlagsFromPipelineStageFlags(stageMask));

        return true;
    }

    bool waitEvents(uint32_t eventCount, event_t*const *const pEvents, const SDependencyInfo* depInfos) override
    {
        constexpr uint32_t MAX_EVENT_COUNT = (1u << 12) / sizeof(VkEvent);
        assert(eventCount <= MAX_EVENT_COUNT);

        constexpr uint32_t MAX_BARRIER_COUNT = 100u;
        assert(depInfos->memBarrierCount <= MAX_BARRIER_COUNT);
        assert(depInfos->bufBarrierCount <= MAX_BARRIER_COUNT);
        assert(depInfos->imgBarrierCount <= MAX_BARRIER_COUNT);

        uint32_t totalResourceCount = 0u;
        core::smart_refctd_ptr<const core::IReferenceCounted> tmp[2 * MAX_BARRIER_COUNT + MAX_EVENT_COUNT];
        {
            uint32_t offset = totalResourceCount;
            uint32_t resourceCount = 0u;
            for (; resourceCount < depInfos->bufBarrierCount; ++resourceCount)
                tmp[offset + resourceCount] = depInfos->bufBarriers[resourceCount].buffer;
            totalResourceCount += resourceCount;
        }
        {
            uint32_t offset = totalResourceCount;
            uint32_t resourceCount = 0u;
            for (; resourceCount < depInfos->imgBarrierCount; ++resourceCount)
                tmp[offset + resourceCount] = depInfos->imgBarriers[resourceCount].image;
            totalResourceCount += resourceCount;
        }
        {
            uint32_t offset = totalResourceCount;
            uint32_t resourceCount = 0u;
            for (; resourceCount < eventCount; ++resourceCount)
                tmp[offset + resourceCount] = core::smart_refctd_ptr<const core::IReferenceCounted>(pEvents[resourceCount]);
            totalResourceCount += resourceCount;
        }

        if (!saveReferencesToResources(tmp, tmp + totalResourceCount))
            return false;

        VkEvent vk_events[MAX_EVENT_COUNT];
        for (uint32_t i = 0u; i < eventCount; ++i)
        {
            if (pEvents[i]->getAPIType() != EAT_VULKAN)
                continue;

            vk_events[i] = IBackendObject::compatibility_cast<const CVulkanEvent*>(pEvents[i], this)->getInternalObject();
        }

        VkMemoryBarrier vk_memoryBarriers[MAX_BARRIER_COUNT];
        for (uint32_t i = 0u; i < depInfos->memBarrierCount; ++i)
        {
            vk_memoryBarriers[i] = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
            vk_memoryBarriers[i].pNext = nullptr; // must be NULL
            vk_memoryBarriers[i].srcAccessMask = static_cast<VkAccessFlags>(depInfos->memBarriers[i].srcAccessMask.value);
            vk_memoryBarriers[i].dstAccessMask = static_cast<VkAccessFlags>(depInfos->memBarriers[i].dstAccessMask.value);
        }

        VkBufferMemoryBarrier vk_bufferMemoryBarriers[MAX_BARRIER_COUNT];
        for (uint32_t i = 0u; i < depInfos->bufBarrierCount; ++i)
        {
            vk_bufferMemoryBarriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            vk_bufferMemoryBarriers[i].pNext = nullptr; // must be NULL
            vk_bufferMemoryBarriers[i].srcAccessMask = static_cast<VkAccessFlags>(depInfos->bufBarriers[i].barrier.srcAccessMask.value);
            vk_bufferMemoryBarriers[i].dstAccessMask = static_cast<VkAccessFlags>(depInfos->bufBarriers[i].barrier.dstAccessMask.value);
            vk_bufferMemoryBarriers[i].srcQueueFamilyIndex = depInfos->bufBarriers[i].srcQueueFamilyIndex;
            vk_bufferMemoryBarriers[i].dstQueueFamilyIndex = depInfos->bufBarriers[i].dstQueueFamilyIndex;
            vk_bufferMemoryBarriers[i].buffer = IBackendObject::compatibility_cast<const CVulkanBuffer*>(depInfos->bufBarriers[i].buffer.get(), this)->getInternalObject();
            vk_bufferMemoryBarriers[i].offset = depInfos->bufBarriers[i].offset;
            vk_bufferMemoryBarriers[i].size = depInfos->bufBarriers[i].size;
        }

        VkImageMemoryBarrier vk_imageMemoryBarriers[MAX_BARRIER_COUNT];
        for (uint32_t i = 0u; i < depInfos->imgBarrierCount; ++i)
        {
            vk_imageMemoryBarriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            vk_imageMemoryBarriers[i].pNext = nullptr; // pNext must be NULL or a pointer to a valid instance of VkSampleLocationsInfoEXT
            vk_imageMemoryBarriers[i].srcAccessMask = static_cast<VkAccessFlags>(depInfos->imgBarriers[i].barrier.srcAccessMask.value);
            vk_imageMemoryBarriers[i].dstAccessMask = static_cast<VkAccessFlags>(depInfos->imgBarriers[i].barrier.dstAccessMask.value);
            vk_imageMemoryBarriers[i].oldLayout = static_cast<VkImageLayout>(depInfos->imgBarriers[i].oldLayout);
            vk_imageMemoryBarriers[i].newLayout = static_cast<VkImageLayout>(depInfos->imgBarriers[i].newLayout);
            vk_imageMemoryBarriers[i].srcQueueFamilyIndex = depInfos->imgBarriers[i].srcQueueFamilyIndex;
            vk_imageMemoryBarriers[i].dstQueueFamilyIndex = depInfos->imgBarriers[i].dstQueueFamilyIndex;
            vk_imageMemoryBarriers[i].image = IBackendObject::compatibility_cast<const CVulkanImage*>(depInfos->imgBarriers[i].image.get(), this)->getInternalObject();
            vk_imageMemoryBarriers[i].subresourceRange.aspectMask = static_cast<VkImageAspectFlags>(depInfos->imgBarriers[i].subresourceRange.aspectMask);
            vk_imageMemoryBarriers[i].subresourceRange.baseMipLevel = depInfos->imgBarriers[i].subresourceRange.baseMipLevel;
            vk_imageMemoryBarriers[i].subresourceRange.levelCount = depInfos->imgBarriers[i].subresourceRange.levelCount;
            vk_imageMemoryBarriers[i].subresourceRange.baseArrayLayer = depInfos->imgBarriers[i].subresourceRange.baseArrayLayer;
            vk_imageMemoryBarriers[i].subresourceRange.layerCount = depInfos->imgBarriers[i].subresourceRange.layerCount;
        }

        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdWaitEvents(
            m_cmdbuf,
            eventCount,
            vk_events,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, // No way to get this!
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, // No way to get this!
            depInfos->memBarrierCount,
            vk_memoryBarriers,
            depInfos->bufBarrierCount,
            vk_bufferMemoryBarriers,
            depInfos->imgBarrierCount,
            vk_imageMemoryBarriers);

        return true;
    }

    bool pipelineBarrier_impl(core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> srcStageMask,
        core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> dstStageMask,
        core::bitflag<asset::E_DEPENDENCY_FLAGS> dependencyFlags,
        uint32_t memoryBarrierCount, const asset::SMemoryBarrier* pMemoryBarriers,
        uint32_t bufferMemoryBarrierCount, const SBufferMemoryBarrier* pBufferMemoryBarriers,
        uint32_t imageMemoryBarrierCount, const SImageMemoryBarrier* pImageMemoryBarriers) override
    {
        constexpr uint32_t MAX_BARRIER_COUNT = 100u;

        assert(memoryBarrierCount <= MAX_BARRIER_COUNT);
        assert(bufferMemoryBarrierCount <= MAX_BARRIER_COUNT);
        assert(imageMemoryBarrierCount <= MAX_BARRIER_COUNT);

        core::smart_refctd_ptr<const core::IReferenceCounted> tmp[2*MAX_BARRIER_COUNT];

        uint32_t totalResourceCount = 0u;
        for (; totalResourceCount < bufferMemoryBarrierCount; ++totalResourceCount)
            tmp[totalResourceCount] = pBufferMemoryBarriers[totalResourceCount].buffer;

        for (; totalResourceCount < imageMemoryBarrierCount; ++totalResourceCount)
            tmp[totalResourceCount] = pImageMemoryBarriers[totalResourceCount].image;

        // TODO(achal): Remove.
        saveReferencesToResources(tmp, tmp + totalResourceCount);

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
            vk_imageMemoryBarriers[i].subresourceRange.aspectMask = static_cast<VkImageAspectFlags>(pImageMemoryBarriers[i].subresourceRange.aspectMask);
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

    bool beginRenderPass_impl(const SRenderpassBeginInfo* pRenderPassBegin, asset::E_SUBPASS_CONTENTS content) override
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

    bool nextSubpass(asset::E_SUBPASS_CONTENTS contents) override
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdNextSubpass(m_cmdbuf, static_cast<VkSubpassContents>(contents));
        return true;
    }

    bool endRenderPass() override
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdEndRenderPass(m_cmdbuf);
        return true;
    }

    bool setDeviceMask(uint32_t deviceMask) override
    {
        m_deviceMask = deviceMask;
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdSetDeviceMask(m_cmdbuf, deviceMask);
        return true;
    }

    //those two instead of bindPipeline(E_PIPELINE_BIND_POINT, pipeline)
    bool bindGraphicsPipeline(const graphics_pipeline_t* pipeline) override
    {
        if (pipeline->getAPIType() != EAT_VULKAN)
            return false;

        const core::smart_refctd_ptr<const core::IReferenceCounted> tmp[] = { core::smart_refctd_ptr<const graphics_pipeline_t>(pipeline) };
        if (!saveReferencesToResources(tmp, tmp + 1))
            return false;

        VkPipeline vk_pipeline = IBackendObject::compatibility_cast<const CVulkanGraphicsPipeline*>(pipeline, this)->getInternalObject();
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdBindPipeline(m_cmdbuf, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

        return true;
    }

    void bindComputePipeline_impl(const compute_pipeline_t* pipeline) override
    {
        VkPipeline vk_pipeline = IBackendObject::compatibility_cast<const CVulkanComputePipeline*>(pipeline, this)->getInternalObject();
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdBindPipeline(m_cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, vk_pipeline);
    }

    
    bool resetQueryPool(IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount) override;
    bool beginQuery(IQueryPool* queryPool, uint32_t query, core::bitflag<video::IQueryPool::E_QUERY_CONTROL_FLAGS>) override;
    bool endQuery(IQueryPool* queryPool, uint32_t query) override;
    bool copyQueryPoolResults(IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, buffer_t* dstBuffer, size_t dstOffset, size_t stride, core::bitflag<video::IQueryPool::E_QUERY_RESULTS_FLAGS> flags) override;
    bool writeTimestamp(asset::E_PIPELINE_STAGE_FLAGS pipelineStage, IQueryPool* queryPool, uint32_t query) override;

    // Acceleration Structure Properties (Only available on Vulkan)
    bool writeAccelerationStructureProperties(const core::SRange<IGPUAccelerationStructure>& pAccelerationStructures, IQueryPool::E_QUERY_TYPE queryType, IQueryPool* queryPool, uint32_t firstQuery) override;

    bool bindDescriptorSets_impl(asset::E_PIPELINE_BIND_POINT pipelineBindPoint,
        const pipeline_layout_t* layout, uint32_t firstSet, uint32_t descriptorSetCount,
        const descriptor_set_t* const* const pDescriptorSets, 
        const uint32_t dynamicOffsetCount=0u, const uint32_t* dynamicOffsets=nullptr) override
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
                    auto bindings = pDescriptorSets[i]->getLayout()->getBindings();
                    for (const auto& binding : bindings)
                    {
                        if ((binding.type == asset::EDT_STORAGE_BUFFER_DYNAMIC) || (binding.type == asset::EDT_UNIFORM_BUFFER_DYNAMIC))
                            dynamicOffsetCountPerSet[i] += binding.count;
                    }
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
                            firstSet+first, last - first, vk_descriptorSets+first, 0u, nullptr);
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

    bool pushConstants(const pipeline_layout_t* layout, core::bitflag<asset::IShader::E_SHADER_STAGE> stageFlags, uint32_t offset, uint32_t size, const void* pValues) override
    {
        if (layout->getAPIType() != EAT_VULKAN)
            return false;

        const core::smart_refctd_ptr<const core::IReferenceCounted> tmp[] = { core::smart_refctd_ptr<const core::IReferenceCounted>(layout) };
        if (!saveReferencesToResources(tmp, tmp + 1))
            return false;

        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdPushConstants(m_cmdbuf,
            IBackendObject::compatibility_cast<const CVulkanPipelineLayout*>(layout, this)->getInternalObject(),
            static_cast<VkShaderStageFlags>(stageFlags.value),
            offset,
            size,
            pValues);

        return true;
    }

    bool clearColorImage(image_t* image, asset::E_IMAGE_LAYOUT imageLayout, const asset::SClearColorValue* pColor, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges) override
    {
        if (!image || image->getAPIType() != EAT_VULKAN)
            return false;

        const core::smart_refctd_ptr<const core::IReferenceCounted> tmp[] = { core::smart_refctd_ptr<const core::IReferenceCounted>(image) };
        if (!saveReferencesToResources(tmp, tmp + 1))
            return false;

        VkClearColorValue vk_clearColorValue;
        for (uint32_t k = 0u; k < 4u; ++k)
            vk_clearColorValue.uint32[k] = pColor->uint32[k];

        constexpr uint32_t MAX_COUNT = (1u << 12) / sizeof(VkImageSubresourceRange);
        assert(rangeCount <= MAX_COUNT);
        VkImageSubresourceRange vk_ranges[MAX_COUNT];

        for (uint32_t i = 0u; i < rangeCount; ++i)
        {
            vk_ranges[i].aspectMask = static_cast<VkImageAspectFlags>(pRanges[i].aspectMask);
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

    bool clearDepthStencilImage(image_t* image, asset::E_IMAGE_LAYOUT imageLayout, const asset::SClearDepthStencilValue* pDepthStencil, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges) override
    {
        if (!image || image->getAPIType() != EAT_VULKAN)
            return false;

        const core::smart_refctd_ptr<const core::IReferenceCounted> tmp[] = { core::smart_refctd_ptr<const core::IReferenceCounted>(image) };
        if (!saveReferencesToResources(tmp, tmp + 1))
            return false;

        VkClearDepthStencilValue vk_clearDepthStencilValue = { pDepthStencil[0].depth, pDepthStencil[0].stencil };

        constexpr uint32_t MAX_COUNT = (1u << 12) / sizeof(VkImageSubresourceRange);
        assert(rangeCount <= MAX_COUNT);
        VkImageSubresourceRange vk_ranges[MAX_COUNT];

        for (uint32_t i = 0u; i < rangeCount; ++i)
        {
            vk_ranges[i].aspectMask = static_cast<VkImageAspectFlags>(pRanges[i].aspectMask);
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

    bool clearAttachments(uint32_t attachmentCount, const asset::SClearAttachment* pAttachments, uint32_t rectCount, const asset::SClearRect* pRects) override
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

    bool fillBuffer(buffer_t* dstBuffer, size_t dstOffset, size_t size, uint32_t data) override
    {
        if (!dstBuffer || dstBuffer->getAPIType() != EAT_VULKAN)
            return false;

        const core::smart_refctd_ptr<const core::IReferenceCounted> tmp[] = { core::smart_refctd_ptr<const core::IReferenceCounted>(dstBuffer) };
        if (!saveReferencesToResources(tmp, tmp + 1))
            return false;

        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdFillBuffer(
            m_cmdbuf,
            IBackendObject::compatibility_cast<const CVulkanBuffer*>(dstBuffer, this)->getInternalObject(),
            static_cast<VkDeviceSize>(dstOffset),
            static_cast<VkDeviceSize>(size),
            data);

        return true;
    }

    bool updateBuffer(buffer_t* dstBuffer, size_t dstOffset, size_t dataSize, const void* pData) override
    {
        if (!dstBuffer || dstBuffer->getAPIType() != EAT_VULKAN)
            return false;

        const core::smart_refctd_ptr<const core::IReferenceCounted> tmp[] = { core::smart_refctd_ptr<const core::IReferenceCounted>(dstBuffer) };
        if (!saveReferencesToResources(tmp, tmp + 1))
            return false;

        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdUpdateBuffer(
            m_cmdbuf,
            IBackendObject::compatibility_cast<const CVulkanBuffer*>(dstBuffer, this)->getInternalObject(),
            static_cast<VkDeviceSize>(dstOffset),
            static_cast<VkDeviceSize>(dataSize),
            pData);

        return true;
    }

    bool executeCommands(uint32_t count, cmdbuf_t* const* const cmdbufs) override
    {
        constexpr uint32_t MAX_COMMAND_BUFFER_COUNT = (1ull << 12)/sizeof(void*);
        assert(count <= MAX_COMMAND_BUFFER_COUNT);

        core::smart_refctd_ptr<const core::IReferenceCounted> tmp[MAX_COMMAND_BUFFER_COUNT] = {};
        VkCommandBuffer vk_commandBuffers[MAX_COMMAND_BUFFER_COUNT];

        for (uint32_t i = 0u; i < count; ++i)
        {
            if (!cmdbufs[i] || cmdbufs[i]->getAPIType() != EAT_VULKAN || cmdbufs[i]->getLevel() != EL_SECONDARY)
                return false;

            tmp[i] = core::smart_refctd_ptr<const IGPUCommandBuffer>(cmdbufs[i]);

            vk_commandBuffers[i] = IBackendObject::compatibility_cast<const CVulkanCommandBuffer*>(cmdbufs[i], this)->getInternalObject();
        }

        if (!saveReferencesToResources(tmp, tmp + count))
            return false;

        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdExecuteCommands(m_cmdbuf, count, vk_commandBuffers);

        return true;
    }

    bool buildAccelerationStructures(const core::SRange<IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, IGPUAccelerationStructure::BuildRangeInfo* const* ppBuildRangeInfos) override;
    
    bool buildAccelerationStructuresIndirect(
        const core::SRange<IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, 
        const core::SRange<IGPUAccelerationStructure::DeviceAddressType>& pIndirectDeviceAddresses,
        const uint32_t* pIndirectStrides,
        const uint32_t* const* ppMaxPrimitiveCounts) override;

    bool copyAccelerationStructure(const IGPUAccelerationStructure::CopyInfo& copyInfo) override;
    
    bool copyAccelerationStructureToMemory(const IGPUAccelerationStructure::DeviceCopyToMemoryInfo& copyInfo) override;

    bool copyAccelerationStructureFromMemory(const IGPUAccelerationStructure::DeviceCopyFromMemoryInfo& copyInfo) override;
    
	inline const void* getNativeHandle() const override {return &m_cmdbuf;}
    VkCommandBuffer getInternalObject() const {return m_cmdbuf;}

private:
    void freeSpaceInCmdPool()
    {
        if (m_cmdpool->getAPIType() == EAT_VULKAN && m_argListHead)
        {
            CVulkanCommandPool* vulkanCommandPool = IBackendObject::compatibility_cast<CVulkanCommandPool*>(m_cmdpool.get(), this);
            vulkanCommandPool->free_all(m_argListHead);
            m_argListHead = nullptr;
            m_argListTail = nullptr;
        }
    }

    bool saveReferencesToResources(const core::smart_refctd_ptr<const core::IReferenceCounted>* begin,
        const core::smart_refctd_ptr<const core::IReferenceCounted>* end)
    {
        if (m_cmdpool->getAPIType() != EAT_VULKAN)
            return false;

        CVulkanCommandPool* vulkanCommandPool = IBackendObject::compatibility_cast<CVulkanCommandPool*>(m_cmdpool.get(), this);
        vulkanCommandPool->emplace_n(m_argListTail, begin, end);

        return true;
    }

    CVulkanCommandPool::ArgumentReferenceSegment* m_argListHead = nullptr;
    CVulkanCommandPool::ArgumentReferenceSegment* m_argListTail = nullptr;
    VkCommandBuffer m_cmdbuf;
};  

}

#endif
