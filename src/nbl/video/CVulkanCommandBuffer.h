#ifndef __NBL_C_VULKAN_COMMAND_BUFFER_H_INCLUDED__
#define __NBL_C_VULKAN_COMMAND_BUFFER_H_INCLUDED__

#include "nbl/video/IGPUCommandBuffer.h"

// Todo(achal): I think I a lot of them could be made forward declarations if I introduce
// a CVulkanCommandBuffer.cpp
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
        // Should we do manual state management? Wouldn't relying on validation errors be better?
        // IGPUCommandBuffer::begin(recordingFlags);
        
        VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.pNext = nullptr; // pNext must be NULL or a pointer to a valid instance of VkDeviceGroupCommandBufferBeginInfo
        beginInfo.flags = static_cast<VkCommandBufferUsageFlags>(recordingFlags);
        beginInfo.pInheritanceInfo = nullptr; // useful if it was a secondary command buffer
        
        assert(vkBeginCommandBuffer(m_cmdbuf, &beginInfo) == VK_SUCCESS);
    }

    // API needs to changed, vkEndCommandBuffer can fail
    void end() override
    {
        assert(vkEndCommandBuffer(m_cmdbuf) == VK_SUCCESS);
    }

    bool reset(uint32_t _flags) override
    {
        freeSpaceInCmdPool();

        if (vkResetCommandBuffer(m_cmdbuf, static_cast<VkCommandBufferResetFlags>(_flags)) == VK_SUCCESS)
            return true;
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
        return false;
    }

    bool copyImage(const image_t* srcImage, asset::E_IMAGE_LAYOUT srcImageLayout, image_t* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SImageCopy* pRegions) override
    {
        return false;
    }

    bool copyBufferToImage(const buffer_t* srcBuffer, image_t* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) override
    {
        return false;
    }

    bool copyImageToBuffer(const image_t* srcImage, asset::E_IMAGE_LAYOUT srcImageLayout, buffer_t* dstBuffer, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) override
    {
        return false;
    }

    bool blitImage(const image_t* srcImage, asset::E_IMAGE_LAYOUT srcImageLayout, image_t* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageBlit* pRegions, asset::ISampler::E_TEXTURE_FILTER filter) override
    {
        return false;
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

    bool pipelineBarrier(std::underlying_type_t<asset::E_PIPELINE_STAGE_FLAGS> srcStageMask,
        std::underlying_type_t<asset::E_PIPELINE_STAGE_FLAGS> dstStageMask,
        std::underlying_type_t<asset::E_DEPENDENCY_FLAGS> dependencyFlags,
        uint32_t memoryBarrierCount, const asset::SMemoryBarrier* pMemoryBarriers,
        uint32_t bufferMemoryBarrierCount, const SBufferMemoryBarrier* pBufferMemoryBarriers,
        uint32_t imageMemoryBarrierCount, const SImageMemoryBarrier* pImageMemoryBarriers) override
    {
        // Todo(achal): I could probably abstract this out into something like getVulkanCommandPool

#if 1
        assert(memoryBarrierCount <= 100);
        VkMemoryBarrier vk_memoryBarriers[100];
        for (uint32_t i = 0u; i < memoryBarrierCount; ++i)
        {
            vk_memoryBarriers[i] = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
            vk_memoryBarriers[i].pNext = nullptr; // must be NULL
            vk_memoryBarriers[i].srcAccessMask = static_cast<VkAccessFlags>(pMemoryBarriers[i].srcAccessMask);
            vk_memoryBarriers[i].dstAccessMask = static_cast<VkAccessFlags>(pMemoryBarriers[i].dstAccessMask);
        }

        // Todo(achal): Proper lifetime management of IGPUBuffer
        assert(bufferMemoryBarrierCount <= 100);
        VkBufferMemoryBarrier vk_bufferMemoryBarriers[100];
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

        // Todo(achal): Proper lifetime management of IGPUImage
        assert(imageMemoryBarrierCount <= 100);
        VkImageMemoryBarrier vk_imageMemoryBarriers[100];
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

        vkCmdPipelineBarrier(m_cmdbuf, static_cast<VkPipelineStageFlags>(srcStageMask),
            static_cast<VkPipelineStageFlags>(dstStageMask),
            static_cast<VkDependencyFlags>(dependencyFlags),
            memoryBarrierCount, vk_memoryBarriers,
            bufferMemoryBarrierCount, vk_bufferMemoryBarriers,
            imageMemoryBarrierCount, vk_imageMemoryBarriers);

        return true;
#else
        return false;
#endif
    }

    bool beginRenderPass(const SRenderpassBeginInfo* pRenderPassBegin, asset::E_SUBPASS_CONTENTS content) override
    {
        return false;
    }

    bool nextSubpass(asset::E_SUBPASS_CONTENTS contents) override
    {
        return false;
    }

    bool endRenderPass() override
    {
        return false;
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
        if (pipeline->getAPIType() != EAT_VULKAN)
            return false;

        if (m_cmdpool->getAPIType() != EAT_VULKAN)
            return false;

        const core::smart_refctd_ptr<const core::IReferenceCounted> tmp[] = 
            { core::smart_refctd_ptr<const compute_pipeline_t>(pipeline) };

        CVulkanCommandPool* vulkanCommandPool = static_cast<CVulkanCommandPool*>(m_cmdpool.get());
        vulkanCommandPool->emplace_n(m_argListTail, tmp, tmp + 1u);

        VkPipeline vk_pipeline = static_cast<const CVulkanComputePipeline*>(pipeline)->getInternalObject();
        vkCmdBindPipeline(m_cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, vk_pipeline);

        return true;
    }

    //virtual bool resetQueryPool(IGPUQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount) = 0;
    //virtual bool beginQuery(IGPUQueryPool* queryPool, uint32_t entry, std::underlying_type_t<E_QUERY_CONTROL_FLAGS> flags) = 0;
    //virtual bool endQuery(IGPUQueryPool* queryPool, uint32_t query) = 0;
    //virtual bool copyQueryPoolResults(IGPUQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, buffer_t* dstBuffer, size_t dstOffset, size_t stride, std::underlying_type_t<E_QUERY_RESULT_FLAGS> flags) = 0;
    //virtual bool writeTimestamp(std::underlying_type_t<asset::E_PIPELINE_STAGE_FLAGS> pipelineStage, IGPUQueryPool* queryPool, uint32_t query) = 0;

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

    bool pushConstants(const pipeline_layout_t* layout, std::underlying_type_t<asset::ISpecializedShader::E_SHADER_STAGE> stageFlags, uint32_t offset, uint32_t size, const void* pValues) override
    {
        return false;
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

    ArgumentReferenceSegment* m_argListHead = nullptr;
    ArgumentReferenceSegment* m_argListTail = nullptr;
    VkCommandBuffer m_cmdbuf;
};

}

#endif
