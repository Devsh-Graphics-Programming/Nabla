#ifndef __NBL_C_VULKAN_COMMAND_BUFFER_H_INCLUDED__
#define __NBL_C_VULKAN_COMMAND_BUFFER_H_INCLUDED__

#include "nbl/video/IGPUCommandBuffer.h"
// Todo(achal): I think I a lot of them could be made forward declarations if I introduce
// a CVulkanCommandBuffer.cpp
#include "nbl/video/CVulkanCommandPool.h"
#include "nbl/video/CVulkanBuffer.h"
#include "nbl/video/CVulkanImage.h"
#include "nbl/video/CVulkanComputePipeline.h"
#include "nbl/video/CVulkanPipelineLayout.h"
#include "nbl/video/CVulkanDescriptorSet.h"

#include <volk.h>

namespace nbl::video
{
struct ArgumentReferenceSegment;

class CVulkanCommandBuffer : public IGPUCommandBuffer
{
public:
    CVulkanCommandBuffer(core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, E_LEVEL level,
        VkCommandBuffer _vkcmdbuf, IGPUCommandPool* commandPool)
        : IGPUCommandBuffer(std::move(logicalDevice), level, commandPool), m_cmdbuf(_vkcmdbuf)
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

    // API needs to change, vkBeginCommandBuffer can fail
    void begin(uint32_t recordingFlags) override
    {
        VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.pNext = nullptr; // pNext must be NULL or a pointer to a valid instance of VkDeviceGroupCommandBufferBeginInfo
        beginInfo.flags = static_cast<VkCommandBufferUsageFlags>(recordingFlags);
        beginInfo.pInheritanceInfo = nullptr; // useful if it was a secondary command buffer
        
        VkResult retval = vkBeginCommandBuffer(m_cmdbuf, &beginInfo);
        assert(retval == VK_SUCCESS);
        IGPUCommandBuffer::begin(recordingFlags);
    }

    // API needs to changed, vkEndCommandBuffer can fail
    void end() override
    {
        VkResult retval = vkEndCommandBuffer(m_cmdbuf);
        assert(retval == VK_SUCCESS);
        IGPUCommandBuffer::end();
    }

    bool reset(uint32_t _flags) override
    {
        freeSpaceInCmdPool();

        if (vkResetCommandBuffer(m_cmdbuf, static_cast<VkCommandBufferResetFlags>(_flags)) == VK_SUCCESS)
        {
            IGPUCommandBuffer::reset(_flags);
            return true;
        }
        else
            return false;
    }

    virtual bool bindIndexBuffer(const buffer_t* buffer, size_t offset, asset::E_INDEX_TYPE indexType) override
    {
        return false;
    }

    bool draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex,
        uint32_t firstInstance) override
    {
        return false;
    }

    bool drawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex,
        int32_t vertexOffset, uint32_t firstInstance) override
    {
        return false;
    }

    bool drawIndirect(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride) override
    {
        return false;
    }
    bool drawIndexedIndirect(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride) override
    {
        return false;
    }

    bool drawIndirectCount(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride) override
    {
        return false;
    }
    bool drawIndexedIndirectCount(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride) override
    {
        return false;
    }

    bool drawMeshBuffer(const nbl::video::IGPUMeshBuffer* meshBuffer) override
    {
        return false;
    }

    bool setViewport(uint32_t firstViewport, uint32_t viewportCount,
        const asset::SViewport* pViewports) override
    {
        return false;
    }

    bool setLineWidth(float lineWidth) override
    {
        return false;
    }

    bool setDepthBias(float depthBiasConstantFactor, float depthBiasClamp, float depthBiasSlopeFactor) override
    {
        return false;
    }

    bool setBlendConstants(const float blendConstants[4]) override
    {
        return false;
    }

    bool copyBuffer(const buffer_t* srcBuffer, buffer_t* dstBuffer, uint32_t regionCount, const asset::SBufferCopy* pRegions) override
    {
        const core::smart_refctd_ptr<const core::IReferenceCounted> tmp[2] = {
            core::smart_refctd_ptr<const IGPUBuffer>(srcBuffer),
            core::smart_refctd_ptr<const IGPUBuffer>(dstBuffer) };

        if (!saveReferencesToResources(tmp, tmp + 2))
            return false;

        if ((srcBuffer->getAPIType() != EAT_VULKAN) || (dstBuffer->getAPIType() != EAT_VULKAN))
            return false;

        VkBuffer vk_srcBuffer = static_cast<const CVulkanBuffer*>(srcBuffer)->getInternalObject();
        VkBuffer vk_dstBuffer = static_cast<const CVulkanBuffer*>(dstBuffer)->getInternalObject();

        constexpr uint32_t MAX_BUFFER_COPY_REGION_COUNT = 681u;
        VkBufferCopy vk_bufferCopyRegions[MAX_BUFFER_COPY_REGION_COUNT];
        for (uint32_t i = 0u; i < regionCount; ++i)
        {
            vk_bufferCopyRegions[i].srcOffset = pRegions[i].srcOffset;
            vk_bufferCopyRegions[i].dstOffset = pRegions[i].dstOffset;
            vk_bufferCopyRegions[i].size = pRegions[i].size;
        }

        vkCmdCopyBuffer(m_cmdbuf, vk_srcBuffer, vk_dstBuffer, regionCount, vk_bufferCopyRegions);

        return true;
    }

    bool copyImage(const image_t* srcImage, asset::E_IMAGE_LAYOUT srcImageLayout, image_t* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SImageCopy* pRegions) override
    {
        return false;
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
            vk_regions[i].imageSubresource.aspectMask = pRegions[i].imageSubresource.aspectMask;
            vk_regions[i].imageSubresource.mipLevel = pRegions[i].imageSubresource.mipLevel;
            vk_regions[i].imageSubresource.baseArrayLayer = pRegions[i].imageSubresource.baseArrayLayer;
            vk_regions[i].imageSubresource.layerCount = pRegions[i].imageSubresource.layerCount;
            vk_regions[i].imageOffset = { static_cast<int32_t>(pRegions[i].imageOffset.x), static_cast<int32_t>(pRegions[i].imageOffset.y), static_cast<int32_t>(pRegions[i].imageOffset.z) }; // Todo(achal): Make the regular old assignment operator work
            vk_regions[i].imageExtent = { pRegions[i].imageExtent.width, pRegions[i].imageExtent.height, pRegions[i].imageExtent.depth }; // Todo(achal): Make the regular old assignment operator work
        }

        vkCmdCopyBufferToImage(m_cmdbuf,
            static_cast<const video::CVulkanBuffer*>(srcBuffer)->getInternalObject(),
            static_cast<const video::CVulkanImage*>(dstImage)->getInternalObject(),
            static_cast<VkImageLayout>(dstImageLayout), regionCount, vk_regions);

        return true;
    }

    bool copyImageToBuffer(const image_t* srcImage, asset::E_IMAGE_LAYOUT srcImageLayout, buffer_t* dstBuffer, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) override
    {
        return false;
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

        VkImage vk_srcImage = static_cast<const CVulkanImage*>(srcImage)->getInternalObject();
        VkImage vk_dstImage = static_cast<const CVulkanImage*>(dstImage)->getInternalObject();

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

        vkCmdBlitImage(m_cmdbuf, vk_srcImage, static_cast<VkImageLayout>(srcImageLayout),
            vk_dstImage, static_cast<VkImageLayout>(dstImageLayout), regionCount, vk_blitRegions,
            static_cast<VkFilter>(filter));

        return true;
    }

    bool resolveImage(const image_t* srcImage, asset::E_IMAGE_LAYOUT srcImageLayout, image_t* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageResolve* pRegions) override
    {
        return false;
    }

    bool bindVertexBuffers(uint32_t firstBinding, uint32_t bindingCount, const buffer_t* const *const pBuffers, const size_t* pOffsets) override
    {
        return false;
    }

    bool setScissor(uint32_t firstScissor, uint32_t scissorCount, const VkRect2D* pScissors) override
    {
        return false;
    }

    bool setDepthBounds(float minDepthBounds, float maxDepthBounds) override
    {
        return false;
    }

    bool setStencilCompareMask(asset::E_STENCIL_FACE_FLAGS faceMask, uint32_t compareMask) override
    {
        return false;
    }

    bool setStencilWriteMask(asset::E_STENCIL_FACE_FLAGS faceMask, uint32_t writeMask) override
    {
        return false;
    }

    bool setStencilReference(asset::E_STENCIL_FACE_FLAGS faceMask, uint32_t reference) override
    {
        return false;
    }

    // Doesn't really require the return value here
    bool dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) override
    {
        vkCmdDispatch(m_cmdbuf, groupCountX, groupCountY, groupCountZ);
        return true;
    }

    bool dispatchIndirect(const buffer_t* buffer, size_t offset) override
    {
        return false;
    }

    bool dispatchBase(uint32_t baseGroupX, uint32_t baseGroupY, uint32_t baseGroupZ, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) override
    {
        return false;
    }

    bool setEvent(event_t* event, const SDependencyInfo& depInfo) override
    {
        return false;
    }

    bool resetEvent(event_t* event, asset::E_PIPELINE_STAGE_FLAGS stageMask) override
    {
        return false;
    }

    bool waitEvents(uint32_t eventCount, event_t*const *const pEvents, const SDependencyInfo* depInfos) override
    {
        return false;
    }

    bool pipelineBarrier(core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> srcStageMask,
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

        if (!saveReferencesToResources(tmp, tmp + totalResourceCount))
            return false;

        VkMemoryBarrier vk_memoryBarriers[MAX_BARRIER_COUNT];
        for (uint32_t i = 0u; i < memoryBarrierCount; ++i)
        {
            vk_memoryBarriers[i] = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
            vk_memoryBarriers[i].pNext = nullptr; // must be NULL
            vk_memoryBarriers[i].srcAccessMask = static_cast<VkAccessFlags>(pMemoryBarriers[i].srcAccessMask);
            vk_memoryBarriers[i].dstAccessMask = static_cast<VkAccessFlags>(pMemoryBarriers[i].dstAccessMask);
        }

        VkBufferMemoryBarrier vk_bufferMemoryBarriers[MAX_BARRIER_COUNT];
        for (uint32_t i = 0u; i < bufferMemoryBarrierCount; ++i)
        {
            vk_bufferMemoryBarriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            vk_bufferMemoryBarriers[i].pNext = nullptr; // must be NULL
            vk_bufferMemoryBarriers[i].srcAccessMask = static_cast<VkAccessFlags>(pBufferMemoryBarriers[i].barrier.srcAccessMask);
            vk_bufferMemoryBarriers[i].dstAccessMask = static_cast<VkAccessFlags>(pBufferMemoryBarriers[i].barrier.dstAccessMask);
            vk_bufferMemoryBarriers[i].srcQueueFamilyIndex = pBufferMemoryBarriers[i].srcQueueFamilyIndex;
            vk_bufferMemoryBarriers[i].dstQueueFamilyIndex = pBufferMemoryBarriers[i].dstQueueFamilyIndex;
            vk_bufferMemoryBarriers[i].buffer = static_cast<const CVulkanBuffer*>(pBufferMemoryBarriers[i].buffer.get())->getInternalObject();
            vk_bufferMemoryBarriers[i].offset = pBufferMemoryBarriers[i].offset;
            vk_bufferMemoryBarriers[i].size = pBufferMemoryBarriers[i].size;
        }

        VkImageMemoryBarrier vk_imageMemoryBarriers[MAX_BARRIER_COUNT];
        for (uint32_t i = 0u; i < imageMemoryBarrierCount; ++i)
        {
            vk_imageMemoryBarriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            vk_imageMemoryBarriers[i].pNext = nullptr; // pNext must be NULL or a pointer to a valid instance of VkSampleLocationsInfoEXT
            vk_imageMemoryBarriers[i].srcAccessMask = static_cast<VkAccessFlags>(pImageMemoryBarriers[i].barrier.srcAccessMask);
            vk_imageMemoryBarriers[i].dstAccessMask = static_cast<VkAccessFlags>(pImageMemoryBarriers[i].barrier.dstAccessMask);
            vk_imageMemoryBarriers[i].oldLayout = static_cast<VkImageLayout>(pImageMemoryBarriers[i].oldLayout);
            vk_imageMemoryBarriers[i].newLayout = static_cast<VkImageLayout>(pImageMemoryBarriers[i].newLayout);
            vk_imageMemoryBarriers[i].srcQueueFamilyIndex = pImageMemoryBarriers[i].srcQueueFamilyIndex;
            vk_imageMemoryBarriers[i].dstQueueFamilyIndex = pImageMemoryBarriers[i].dstQueueFamilyIndex;
            vk_imageMemoryBarriers[i].image = static_cast<const CVulkanImage*>(pImageMemoryBarriers[i].image.get())->getInternalObject();
            vk_imageMemoryBarriers[i].subresourceRange.aspectMask = static_cast<VkImageAspectFlags>(pImageMemoryBarriers[i].subresourceRange.aspectMask);
            vk_imageMemoryBarriers[i].subresourceRange.baseMipLevel = pImageMemoryBarriers[i].subresourceRange.baseMipLevel;
            vk_imageMemoryBarriers[i].subresourceRange.levelCount = pImageMemoryBarriers[i].subresourceRange.levelCount;
            vk_imageMemoryBarriers[i].subresourceRange.baseArrayLayer = pImageMemoryBarriers[i].subresourceRange.baseArrayLayer;
            vk_imageMemoryBarriers[i].subresourceRange.layerCount = pImageMemoryBarriers[i].subresourceRange.layerCount;
        }

        vkCmdPipelineBarrier(m_cmdbuf, static_cast<VkPipelineStageFlags>(srcStageMask.value),
            static_cast<VkPipelineStageFlags>(dstStageMask.value),
            static_cast<VkDependencyFlags>(dependencyFlags.value),
            memoryBarrierCount, vk_memoryBarriers,
            bufferMemoryBarrierCount, vk_bufferMemoryBarriers,
            imageMemoryBarrierCount, vk_imageMemoryBarriers);

        return true;
    }

    bool beginRenderPass(const SRenderpassBeginInfo* pRenderPassBegin, asset::E_SUBPASS_CONTENTS content) override
    {
        if ((pRenderPassBegin->renderpass->getAPIType() != EAT_VULKAN) || (pRenderPassBegin->framebuffer->getAPIType() != EAT_VULKAN))
            return false;

        constexpr uint32_t MAX_CLEAR_VALUE_COUNT = (1 << 12ull) / sizeof(VkClearValue);
        VkClearValue vk_clearValues[MAX_CLEAR_VALUE_COUNT];
        assert(pRenderPassBegin->clearValueCount <= MAX_CLEAR_VALUE_COUNT);

        for (uint32_t i = 0u; i < pRenderPassBegin->clearValueCount; ++i)
        {
            for (uint32_t k = 0u; k < 4u; ++k)
            {
                vk_clearValues[i].color.float32[k] = pRenderPassBegin->clearValues[i].color.float32[k];
                vk_clearValues[i].color.int32[k] = pRenderPassBegin->clearValues[i].color.int32[k];
                vk_clearValues[i].color.uint32[k] = pRenderPassBegin->clearValues[i].color.uint32[k];
            }

            vk_clearValues[i].depthStencil.depth = pRenderPassBegin->clearValues[i].depthStencil.depth;
            vk_clearValues[i].depthStencil.stencil = pRenderPassBegin->clearValues[i].depthStencil.stencil;
        }

        VkRenderPassBeginInfo vk_beginInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
        vk_beginInfo.pNext = nullptr;
        vk_beginInfo.renderPass = static_cast<const CVulkanRenderpass*>(pRenderPassBegin->renderpass.get())->getInternalObject();
        vk_beginInfo.framebuffer = static_cast<const CVulkanFramebuffer*>(pRenderPassBegin->framebuffer.get())->getInternalObject();
        vk_beginInfo.renderArea = pRenderPassBegin->renderArea;
        vk_beginInfo.clearValueCount = pRenderPassBegin->clearValueCount;
        vk_beginInfo.pClearValues = vk_clearValues;

        vkCmdBeginRenderPass(m_cmdbuf, &vk_beginInfo, static_cast<VkSubpassContents>(content));

        return true;
    }

    bool nextSubpass(asset::E_SUBPASS_CONTENTS contents) override
    {
        return false;
    }

    bool endRenderPass() override
    {
        vkCmdEndRenderPass(m_cmdbuf);
        return true;
    }

    bool setDeviceMask(uint32_t deviceMask) override
    {
        // m_deviceMask = deviceMask;
        // return true;
        return false;
    }

    //those two instead of bindPipeline(E_PIPELINE_BIND_POINT, pipeline)
    bool bindGraphicsPipeline(const graphics_pipeline_t* pipeline) override
    {
        return false;
    }

    bool bindComputePipeline(const compute_pipeline_t* pipeline) override
    {
        const core::smart_refctd_ptr<const core::IReferenceCounted> tmp[] = { core::smart_refctd_ptr<const compute_pipeline_t>(pipeline) };
        if (!saveReferencesToResources(tmp, tmp + 1))
            return false;

        if (pipeline->getAPIType() != EAT_VULKAN)
            return false;

        VkPipeline vk_pipeline = static_cast<const CVulkanComputePipeline*>(pipeline)->getInternalObject();
        vkCmdBindPipeline(m_cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, vk_pipeline);

        return true;
    }

    
    bool resetQueryPool(IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount) override;
    bool beginQuery(IQueryPool* queryPool, uint32_t query, IQueryPool::E_QUERY_CONTROL_FLAGS flags = static_cast<IQueryPool::E_QUERY_CONTROL_FLAGS>(0)) override;
    bool endQuery(IQueryPool* queryPool, uint32_t query) override;
    bool copyQueryPoolResults(IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, buffer_t* dstBuffer, size_t dstOffset, size_t stride, IQueryPool::E_QUERY_RESULTS_FLAGS flags) override;
    bool writeTimestamp(asset::E_PIPELINE_STAGE_FLAGS pipelineStage, IQueryPool* queryPool, uint32_t query) override;
    // TRANSFORM_FEEDBACK_STREAM
    bool beginQueryIndexed(IQueryPool* queryPool, uint32_t query, uint32_t index, IQueryPool::E_QUERY_CONTROL_FLAGS flags = static_cast<IQueryPool::E_QUERY_CONTROL_FLAGS>(0)) override;
    bool endQueryIndexed(IQueryPool* queryPool, uint32_t query, uint32_t index) override;
    // Acceleration Structure Properties (Only available on Vulkan)
    bool writeAccelerationStructureProperties(const core::SRange<IGPUAccelerationStructure>& pAccelerationStructures, IQueryPool::E_QUERY_TYPE queryType, IQueryPool* queryPool, uint32_t firstQuery) override;


    // E_PIPELINE_BIND_POINT needs to be in asset namespace or divide this into two functions (for graphics and compute)
    bool bindDescriptorSets(asset::E_PIPELINE_BIND_POINT pipelineBindPoint,
        const pipeline_layout_t* layout, uint32_t firstSet, uint32_t descriptorSetCount,
        const descriptor_set_t* const* const pDescriptorSets,
        core::smart_refctd_dynamic_array<uint32_t> dynamicOffsets = nullptr) override
    {
        if (layout->getAPIType() != EAT_VULKAN)
            return false;

        constexpr uint32_t MAX_DESCRIPTOR_SET_COUNT = 100u;

        VkPipelineLayout vk_pipelineLayout = static_cast<const CVulkanPipelineLayout*>(layout)->getInternalObject();

        VkDescriptorSet vk_descriptorSets[MAX_DESCRIPTOR_SET_COUNT];
        for (uint32_t i = 0u; i < descriptorSetCount; ++i)
        {
            if (pDescriptorSets[i]->getAPIType() == EAT_VULKAN)
                vk_descriptorSets[i] = static_cast<const CVulkanDescriptorSet*>(pDescriptorSets[i])->getInternalObject();
        }

        uint32_t vk_dynamicOffsetCount = 0u;
        uint32_t* vk_dynamicOffsets = nullptr;
        if (dynamicOffsets)
        {
            vk_dynamicOffsetCount = dynamicOffsets->size();
            vk_dynamicOffsets = dynamicOffsets->begin();
        }

        vkCmdBindDescriptorSets(m_cmdbuf, static_cast<VkPipelineBindPoint>(pipelineBindPoint),
            vk_pipelineLayout, firstSet, descriptorSetCount, vk_descriptorSets, vk_dynamicOffsetCount, vk_dynamicOffsets);

        return true;
    }

    bool pushConstants(const pipeline_layout_t* layout, core::bitflag<asset::ISpecializedShader::E_SHADER_STAGE> stageFlags, uint32_t offset, uint32_t size, const void* pValues) override
    {
        if (layout->getAPIType() != EAT_VULKAN)
            return false;

        const core::smart_refctd_ptr<const core::IReferenceCounted> tmp[] = { core::smart_refctd_ptr<const core::IReferenceCounted>(layout) };
        if (!saveReferencesToResources(tmp, tmp + 1))
            return false;

        vkCmdPushConstants(m_cmdbuf,
            static_cast<const CVulkanPipelineLayout*>(layout)->getInternalObject(),
            static_cast<VkShaderStageFlags>(stageFlags.value),
            offset,
            size,
            pValues);

        return true;
    }

    bool clearColorImage(image_t* image, asset::E_IMAGE_LAYOUT imageLayout, const asset::SClearColorValue* pColor, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges) override
    {
        return false;
    }

    bool clearDepthStencilImage(image_t* image, asset::E_IMAGE_LAYOUT imageLayout, const asset::SClearDepthStencilValue* pDepthStencil, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges) override
    {
        return false;
    }

    bool clearAttachments(uint32_t attachmentCount, const asset::SClearAttachment* pAttachments, uint32_t rectCount, const asset::SClearRect* pRects) override
    {
        return false;
    }

    bool fillBuffer(buffer_t* dstBuffer, size_t dstOffset, size_t size, uint32_t data) override
    {
        return false;
    }

    bool updateBuffer(buffer_t* dstBuffer, size_t dstOffset, size_t dataSize, const void* pData) override
    {
        return false;
    }

    bool regenerateMipmaps(image_view_t* imgview) override
    {
        return false;
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

    VkCommandBuffer getInternalObject() const { return m_cmdbuf; }

private:
    void freeSpaceInCmdPool()
    {
        if (m_cmdpool->getAPIType() == EAT_VULKAN)
        {
            CVulkanCommandPool* vulkanCommandPool = static_cast<CVulkanCommandPool*>(m_cmdpool.get());
            vulkanCommandPool->free_all(m_argListHead);
        }
    }

    bool saveReferencesToResources(const core::smart_refctd_ptr<const core::IReferenceCounted>* begin,
        const core::smart_refctd_ptr<const core::IReferenceCounted>* end)
    {
        if (m_cmdpool->getAPIType() != EAT_VULKAN)
            return false;

        CVulkanCommandPool* vulkanCommandPool = static_cast<CVulkanCommandPool*>(m_cmdpool.get());
        vulkanCommandPool->emplace_n(m_argListTail, begin, end);

        return true;
    }

    ArgumentReferenceSegment* m_argListHead = nullptr;
    ArgumentReferenceSegment* m_argListTail = nullptr;
    VkCommandBuffer m_cmdbuf;
};

}

#endif
