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

class CVulkanCommandBuffer : public IGPUCommandBuffer
{
public:
    CVulkanCommandBuffer(core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, E_LEVEL level,
        VkCommandBuffer _vkcmdbuf, core::smart_refctd_ptr<IGPUCommandPool>&& commandPool, system::logger_opt_smart_ptr&& logger)
        : IGPUCommandBuffer(std::move(logicalDevice), level, std::move(commandPool), std::move(logger)), m_cmdbuf(_vkcmdbuf)
    {}

    bool begin_impl(core::bitflag<E_USAGE> recordingFlags, const SInheritanceInfo* inheritanceInfo) override final;

    inline bool end_impl() override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        VkResult retval = vk->vk.vkEndCommandBuffer(m_cmdbuf);
        return retval == VK_SUCCESS;
    }

    inline bool reset_impl(core::bitflag<E_RESET_FLAGS> flags) override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        const VkResult result = vk->vk.vkResetCommandBuffer(m_cmdbuf, static_cast<VkCommandBufferResetFlags>(flags.value));
        return result == VK_SUCCESS;
    }

    inline void bindIndexBuffer_impl(const buffer_t* buffer, size_t offset, asset::E_INDEX_TYPE indexType) override final
    {
        assert(indexType < asset::EIT_UNKNOWN);

        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();

        vk->vk.vkCmdBindIndexBuffer(
            m_cmdbuf,
            IBackendObject::compatibility_cast<const CVulkanBuffer*>(buffer, this)->getInternalObject(),
            static_cast<VkDeviceSize>(offset),
            static_cast<VkIndexType>(indexType));
    }

    inline bool draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdDraw(m_cmdbuf, vertexCount, instanceCount, firstVertex, firstInstance);
        return true;
    }

    inline bool drawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdDrawIndexed(m_cmdbuf, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
        return true;
    }

    inline bool drawIndirect_impl(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride) override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdDrawIndirect(
            m_cmdbuf,
            IBackendObject::compatibility_cast<const CVulkanBuffer*>(buffer, this)->getInternalObject(),
            static_cast<VkDeviceSize>(offset),
            drawCount,
            stride);
        return true;
    }

    inline bool drawIndexedIndirect_impl(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride) override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdDrawIndexedIndirect(
            m_cmdbuf,
            IBackendObject::compatibility_cast<const CVulkanBuffer*>(buffer, this)->getInternalObject(),
            static_cast<VkDeviceSize>(offset),
            drawCount,
            stride);
        return true;
    }

    inline bool drawIndirectCount_impl(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride) override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdDrawIndirectCount(
            m_cmdbuf,
            IBackendObject::compatibility_cast<const CVulkanBuffer*>(buffer, this)->getInternalObject(),
            static_cast<VkDeviceSize>(offset),
            IBackendObject::compatibility_cast<const CVulkanBuffer*>(countBuffer, this)->getInternalObject(),
            static_cast<VkDeviceSize>(countBufferOffset),
            maxDrawCount,
            stride);
        return true;
    }

    inline bool drawIndexedIndirectCount_impl(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride) override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdDrawIndexedIndirectCount(
            m_cmdbuf,
            IBackendObject::compatibility_cast<const CVulkanBuffer*>(buffer, this)->getInternalObject(),
            static_cast<VkDeviceSize>(offset),
            IBackendObject::compatibility_cast<const CVulkanBuffer*>(countBuffer, this)->getInternalObject(),
            static_cast<VkDeviceSize>(countBufferOffset),
            maxDrawCount,
            stride);
        return true;
    }

    bool setViewport(uint32_t firstViewport, uint32_t viewportCount, const asset::SViewport* pViewports) override final;

    inline bool setLineWidth(float lineWidth) override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdSetLineWidth(m_cmdbuf, lineWidth);
        return true;
    }

    inline bool setDepthBias(float depthBiasConstantFactor, float depthBiasClamp, float depthBiasSlopeFactor) override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdSetDepthBias(m_cmdbuf, depthBiasConstantFactor, depthBiasClamp, depthBiasSlopeFactor);
        return true;
    }

    inline bool setBlendConstants(const float blendConstants[4]) override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdSetBlendConstants(m_cmdbuf, blendConstants);
        return true;
    }

    bool copyBuffer_impl(const buffer_t* srcBuffer, buffer_t* dstBuffer, uint32_t regionCount, const asset::SBufferCopy* pRegions) override final;

    bool copyImage_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SImageCopy* pRegions) override final;

    bool copyBufferToImage_impl(const buffer_t* srcBuffer, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) override final;

    bool copyImageToBuffer_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, buffer_t* dstBuffer, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) override final;

    bool blitImage_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageBlit* pRegions, asset::ISampler::E_TEXTURE_FILTER filter) override final;

    bool resolveImage_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageResolve* pRegions) override final;

    void bindVertexBuffers_impl(uint32_t firstBinding, uint32_t bindingCount, const buffer_t* const* const pBuffers, const size_t* pOffsets) override final;

    inline bool setScissor(uint32_t firstScissor, uint32_t scissorCount, const VkRect2D* pScissors) override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdSetScissor(m_cmdbuf, firstScissor, scissorCount, pScissors);
        return true;
    }

    inline bool setDepthBounds(float minDepthBounds, float maxDepthBounds) override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdSetDepthBounds(m_cmdbuf, minDepthBounds, maxDepthBounds);
        return true;
    }

    inline bool setStencilCompareMask(asset::E_STENCIL_FACE_FLAGS faceMask, uint32_t compareMask) override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdSetStencilCompareMask(m_cmdbuf, static_cast<VkStencilFaceFlags>(faceMask), compareMask);
        return true;
    }

    inline bool setStencilWriteMask(asset::E_STENCIL_FACE_FLAGS faceMask, uint32_t writeMask) override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdSetStencilWriteMask(m_cmdbuf, static_cast<VkStencilFaceFlags>(faceMask), writeMask);
        return true;
    }

    inline bool setStencilReference(asset::E_STENCIL_FACE_FLAGS faceMask, uint32_t reference) override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdSetStencilReference(m_cmdbuf, static_cast<VkStencilFaceFlags>(faceMask), reference);
        return true;
    }

    inline bool dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdDispatch(m_cmdbuf, groupCountX, groupCountY, groupCountZ);
        return true;
    }

    inline bool dispatchIndirect_impl(const buffer_t* buffer, size_t offset) override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdDispatchIndirect(
            m_cmdbuf,
            IBackendObject::compatibility_cast<const CVulkanBuffer*>(buffer, this)->getInternalObject(),
            static_cast<VkDeviceSize>(offset));

        return true;
    }

    inline bool dispatchBase(uint32_t baseGroupX, uint32_t baseGroupY, uint32_t baseGroupZ, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdDispatchBase(m_cmdbuf, baseGroupX, baseGroupY, baseGroupZ, groupCountX, groupCountY, groupCountZ);
        return true;
    }

    inline bool setEvent_impl(event_t* _event, const SDependencyInfo& depInfo) override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdSetEvent(
            m_cmdbuf,
            IBackendObject::compatibility_cast<const CVulkanEvent*>(_event, this)->getInternalObject(),
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT); // No way to get this! SDependencyInfo is unused

        return true;
    }

    inline bool resetEvent_impl(event_t* _event, asset::E_PIPELINE_STAGE_FLAGS stageMask) override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdResetEvent(
            m_cmdbuf,
            IBackendObject::compatibility_cast<const CVulkanEvent*>(_event, this)->getInternalObject(),
            getVkPipelineStageFlagsFromPipelineStageFlags(stageMask));

        return true;
    }

    bool waitEvents_impl(uint32_t eventCount, event_t* const* const pEvents, const SDependencyInfo* depInfo) override final;

    bool pipelineBarrier_impl(core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> srcStageMask,
        core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> dstStageMask,
        core::bitflag<asset::E_DEPENDENCY_FLAGS> dependencyFlags,
        uint32_t memoryBarrierCount, const asset::SMemoryBarrier* pMemoryBarriers,
        uint32_t bufferMemoryBarrierCount, const SBufferMemoryBarrier* pBufferMemoryBarriers,
        uint32_t imageMemoryBarrierCount, const SImageMemoryBarrier* pImageMemoryBarriers) override final;

    bool beginRenderPass_impl(const SRenderpassBeginInfo* pRenderPassBegin, asset::E_SUBPASS_CONTENTS content) override final;

    inline bool nextSubpass(asset::E_SUBPASS_CONTENTS contents) override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdNextSubpass(m_cmdbuf, static_cast<VkSubpassContents>(contents));
        return true;
    }

    inline bool endRenderPass() override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdEndRenderPass(m_cmdbuf);
        return true;
    }

    inline bool setDeviceMask(uint32_t deviceMask) override final
    {
        m_deviceMask = deviceMask;
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdSetDeviceMask(m_cmdbuf, deviceMask);
        return true;
    }

    inline bool bindGraphicsPipeline_impl(const graphics_pipeline_t* pipeline) override final
    {
        VkPipeline vk_pipeline = IBackendObject::compatibility_cast<const CVulkanGraphicsPipeline*>(pipeline, this)->getInternalObject();
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdBindPipeline(m_cmdbuf, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

        return true;
    }

    inline void bindComputePipeline_impl(const compute_pipeline_t* pipeline) override final
    {
        VkPipeline vk_pipeline = IBackendObject::compatibility_cast<const CVulkanComputePipeline*>(pipeline, this)->getInternalObject();
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdBindPipeline(m_cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, vk_pipeline);
    }

    bool resetQueryPool_impl(IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount) override final;
    bool beginQuery_impl(IQueryPool* queryPool, uint32_t query, core::bitflag<video::IQueryPool::E_QUERY_CONTROL_FLAGS>) override final;
    bool endQuery_impl(IQueryPool* queryPool, uint32_t query) override final;
    bool copyQueryPoolResults_impl(IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, buffer_t* dstBuffer, size_t dstOffset, size_t stride, core::bitflag<video::IQueryPool::E_QUERY_RESULTS_FLAGS> flags) override final;
    bool writeTimestamp_impl(asset::E_PIPELINE_STAGE_FLAGS pipelineStage, IQueryPool* queryPool, uint32_t query) override final;

    // Acceleration Structure Properties (Only available on Vulkan)
    bool writeAccelerationStructureProperties_impl(const core::SRange<IGPUAccelerationStructure>& pAccelerationStructures, IQueryPool::E_QUERY_TYPE queryType, IQueryPool* queryPool, uint32_t firstQuery) override final;

    bool bindDescriptorSets_impl(asset::E_PIPELINE_BIND_POINT pipelineBindPoint,
        const pipeline_layout_t* layout, uint32_t firstSet, uint32_t descriptorSetCount,
        const descriptor_set_t* const* const pDescriptorSets,
        const uint32_t dynamicOffsetCount = 0u, const uint32_t* dynamicOffsets = nullptr) override final;

    inline bool pushConstants_impl(const pipeline_layout_t* layout, core::bitflag<asset::IShader::E_SHADER_STAGE> stageFlags, uint32_t offset, uint32_t size, const void* pValues) override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdPushConstants(m_cmdbuf,
            IBackendObject::compatibility_cast<const CVulkanPipelineLayout*>(layout, this)->getInternalObject(),
            static_cast<VkShaderStageFlags>(stageFlags.value),
            offset,
            size,
            pValues);
        return true;
    }

    bool clearColorImage_impl(image_t* image, asset::IImage::E_LAYOUT imageLayout, const asset::SClearColorValue* pColor, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges) override final;

    bool clearDepthStencilImage_impl(image_t* image, asset::IImage::E_LAYOUT imageLayout, const asset::SClearDepthStencilValue* pDepthStencil, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges) override final;

    bool clearAttachments(uint32_t attachmentCount, const asset::SClearAttachment* pAttachments, uint32_t rectCount, const asset::SClearRect* pRects) override final;

    inline bool fillBuffer_impl(buffer_t* dstBuffer, size_t dstOffset, size_t size, uint32_t data) override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdFillBuffer(
            m_cmdbuf,
            IBackendObject::compatibility_cast<const CVulkanBuffer*>(dstBuffer, this)->getInternalObject(),
            static_cast<VkDeviceSize>(dstOffset),
            static_cast<VkDeviceSize>(size),
            data);

        return true;
    }

    inline bool updateBuffer_impl(buffer_t* dstBuffer, size_t dstOffset, size_t dataSize, const void* pData) override final
    {
        const auto* vk = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getFunctionTable();
        vk->vk.vkCmdUpdateBuffer(
            m_cmdbuf,
            IBackendObject::compatibility_cast<const CVulkanBuffer*>(dstBuffer, this)->getInternalObject(),
            static_cast<VkDeviceSize>(dstOffset),
            static_cast<VkDeviceSize>(dataSize),
            pData);

        return true;
    }

    bool executeCommands_impl(uint32_t count, cmdbuf_t* const* const cmdbufs) override final;

    bool buildAccelerationStructures_impl(const core::SRange<IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, IGPUAccelerationStructure::BuildRangeInfo* const* ppBuildRangeInfos) override;    
    bool buildAccelerationStructuresIndirect_impl(const core::SRange<IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, const core::SRange<IGPUAccelerationStructure::DeviceAddressType>& pIndirectDeviceAddresses, const uint32_t* pIndirectStrides, const uint32_t* const* ppMaxPrimitiveCounts) override;
    bool copyAccelerationStructure_impl(const IGPUAccelerationStructure::CopyInfo& copyInfo) override;
    bool copyAccelerationStructureToMemory_impl(const IGPUAccelerationStructure::DeviceCopyToMemoryInfo& copyInfo) override;
    bool copyAccelerationStructureFromMemory_impl(const IGPUAccelerationStructure::DeviceCopyFromMemoryInfo& copyInfo) override;
    
	inline const void* getNativeHandle() const override {return &m_cmdbuf;}
    VkCommandBuffer getInternalObject() const {return m_cmdbuf;}

private:
    VkCommandBuffer m_cmdbuf;
};  

}

#endif
