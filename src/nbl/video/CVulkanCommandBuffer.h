#ifndef _NBL_VIDEO_C_VULKAN_COMMAND_BUFFER_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_COMMAND_BUFFER_H_INCLUDED_

#include "nbl/video/IGPUCommandBuffer.h"

#include "nbl/video/CVulkanDeviceFunctionTable.h"

#include "nbl/video/CVulkanEvent.h"
#include "nbl/video/CVulkanBuffer.h"
#include "nbl/video/CVulkanImage.h"
#include "nbl/video/CVulkanAccelerationStructure.h"
#include "nbl/video/CVulkanDescriptorSet.h"
#include "nbl/video/CVulkanRenderpass.h"
#include "nbl/video/CVulkanFramebuffer.h"
#include "nbl/video/CVulkanPipelineLayout.h"
#include "nbl/video/CVulkanComputePipeline.h"

namespace nbl::video
{

class CVulkanCommandBuffer final : public IGPUCommandBuffer
{
    public:
        CVulkanCommandBuffer(core::smart_refctd_ptr<const ILogicalDevice>&& logicalDevice, const IGPUCommandPool::BUFFER_LEVEL level,
            VkCommandBuffer _vkcmdbuf, core::smart_refctd_ptr<IGPUCommandPool>&& commandPool, system::logger_opt_smart_ptr&& logger)
            : IGPUCommandBuffer(std::move(logicalDevice), level, std::move(commandPool), std::move(logger)), m_cmdbuf(_vkcmdbuf)
        {}

	    inline const void* getNativeHandle() const override {return &m_cmdbuf;}
        VkCommandBuffer getInternalObject() const {return m_cmdbuf;}

    private:
        inline void checkForParentPoolReset_impl() const override {}

        bool begin_impl(const core::bitflag<USAGE> recordingFlags, const SInheritanceInfo* const inheritanceInfo) override;
        inline bool end_impl() override
        {
            const VkResult retval = getFunctionTable().vkEndCommandBuffer(m_cmdbuf);
            return retval==VK_SUCCESS;
        }
        inline bool reset_impl(const core::bitflag<RESET_FLAGS> flags) override
        {
            const VkResult result = getFunctionTable().vkResetCommandBuffer(m_cmdbuf,static_cast<VkCommandBufferResetFlags>(flags.value));
            return result==VK_SUCCESS;
        }

        //bool setDeviceMask_impl(uint32_t deviceMask);

        bool setEvent_impl(IEvent* const _event, const SEventDependencyInfo& depInfo) override;
        bool resetEvent_impl(IEvent* const _event, const core::bitflag<stage_flags_t> stageMask) override;
        bool waitEvents_impl(const uint32_t eventCount, IEvent* const* const pEvents, const SEventDependencyInfo* depInfos) override;
        bool pipelineBarrier_impl(const core::bitflag<asset::E_DEPENDENCY_FLAGS> dependencyFlags, const SPipelineBarrierDependencyInfo& depInfo) override;

        bool fillBuffer_impl(const asset::SBufferRange<IGPUBuffer>& range, const uint32_t data) override;
        bool updateBuffer_impl(const asset::SBufferRange<IGPUBuffer>& range, const void* const pData) override;
        bool copyBuffer_impl(const IGPUBuffer* const srcBuffer, IGPUBuffer* const dstBuffer, const uint32_t regionCount, const video::IGPUCommandBuffer::SBufferCopy* const pRegions) override;

        bool clearColorImage_impl(IGPUImage* const image, const IGPUImage::LAYOUT imageLayout, const SClearColorValue* const pColor, const uint32_t rangeCount, const IGPUImage::SSubresourceRange* const pRanges) override;
        bool clearDepthStencilImage_impl(IGPUImage* const image, const IGPUImage::LAYOUT imageLayout, const SClearDepthStencilValue* const pDepthStencil, const uint32_t rangeCount, const IGPUImage::SSubresourceRange* const pRanges) override;
        bool copyBufferToImage_impl(const IGPUBuffer* const srcBuffer, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const IGPUImage::SBufferCopy* const pRegions) override;
        bool copyImageToBuffer_impl(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, const IGPUBuffer* const dstBuffer, const uint32_t regionCount, const IGPUImage::SBufferCopy* const pRegions) override;
        bool copyImage_impl(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const IGPUImage::SImageCopy* const pRegions) override;

        inline bool buildAccelerationStructures_impl(
            const core::SRange<const IGPUBottomLevelAccelerationStructure::DeviceBuildInfo>& infos,
            const IGPUBottomLevelAccelerationStructure::BuildRangeInfo* const* const ppBuildRangeInfos,
            const uint32_t totalGeometryCount
        ) override
        {
            const auto infoCount = infos.size();
            IGPUCommandPool::StackAllocation<VkAccelerationStructureBuildRangeInfoKHR> vk_buildRangeInfos(m_cmdpool,totalGeometryCount);
            IGPUCommandPool::StackAllocation<const VkAccelerationStructureBuildRangeInfoKHR*> vk_pBuildRangeInfos(m_cmdpool,infoCount);
            if (!vk_buildRangeInfos || !vk_pBuildRangeInfos)
                return false;
            
            // TODO: check for the raytracing feature enabled before wasting memory
            IGPUCommandPool::StackAllocation<VkAccelerationStructureGeometryMotionTrianglesDataNV> vk_vertexMotions(m_cmdpool,totalGeometryCount);
            if (!vk_vertexMotions)
                return false;

            auto out_vk_infos = vk_buildRangeInfos.data();
            for (auto i=0u; i<infoCount; i++)
            {
                vk_pBuildRangeInfos[i] = out_vk_infos;
                getVkASBuildRangeInfos(infos[i].inputCount(),ppBuildRangeInfos[i],out_vk_infos);
            }
            return buildAccelerationStructures_impl_impl<IGPUBottomLevelAccelerationStructure>(infos,vk_pBuildRangeInfos.data(),vk_vertexMotions.data());
        }
        inline bool buildAccelerationStructures_impl(const core::SRange<const IGPUTopLevelAccelerationStructure::DeviceBuildInfo>& infos, const IGPUTopLevelAccelerationStructure::BuildRangeInfo* const pBuildRangeInfos) override
        {
            const auto infoCount = infos.size();
            IGPUCommandPool::StackAllocation<VkAccelerationStructureBuildRangeInfoKHR> vk_buildRangeInfos(m_cmdpool,infoCount);
            IGPUCommandPool::StackAllocation<const VkAccelerationStructureBuildRangeInfoKHR*> vk_pBuildRangeInfos(m_cmdpool,infoCount);
            if (!vk_buildRangeInfos || !vk_pBuildRangeInfos)
                return false;
            
            for (auto i=0u; i<infoCount; i++)
            {
                vk_buildRangeInfos[i] = getVkASBuildRangeInfo(pBuildRangeInfos[i]);
                vk_pBuildRangeInfos[i] = vk_buildRangeInfos.data()+i;
            }
            return buildAccelerationStructures_impl_impl<IGPUTopLevelAccelerationStructure>(infos,vk_pBuildRangeInfos.data());
        }
        template<class AccelerationStructure> requires std::is_base_of_v<IGPUAccelerationStructure,AccelerationStructure>
        inline bool buildAccelerationStructures_impl_impl(
            const core::SRange<const typename AccelerationStructure::DeviceBuildInfo>& infos,
            const VkAccelerationStructureBuildRangeInfoKHR* const* const vk_ppBuildRangeInfos,
            VkAccelerationStructureGeometryMotionTrianglesDataNV* out_vk_vertexMotions=nullptr
        )
        {
            const auto infoCount = infos.size();
            IGPUCommandPool::StackAllocation<VkAccelerationStructureBuildGeometryInfoKHR> vk_buildGeomsInfos(m_cmdpool,infoCount);
            // I can actually rely on this pointer arithmetic because I allocated and populated the arrays myself
            const uint32_t totalGeometryCount = infos[infoCount-1].inputCount()+(vk_ppBuildRangeInfos[infoCount-1]-vk_ppBuildRangeInfos[0]);
            IGPUCommandPool::StackAllocation<VkAccelerationStructureGeometryKHR> vk_geometries(m_cmdpool,totalGeometryCount);
            if (!vk_geometries || !vk_buildGeomsInfos)
                return false;

            auto out_vk_geoms = vk_geometries.data();
            for (auto i=0u; i<infoCount; i++)
                getVkASBuildGeometryInfo<typename AccelerationStructure::DeviceBuildInfo>(infos[i],out_vk_geoms,out_vk_vertexMotions);

            getFunctionTable().vkCmdBuildAccelerationStructuresKHR(m_cmdbuf,infoCount,vk_buildGeomsInfos.data(),vk_ppBuildRangeInfos);
            return true;
        }

        bool buildAccelerationStructuresIndirect_impl(
            const IGPUBuffer* indirectRangeBuffer, const core::SRange<const IGPUBottomLevelAccelerationStructure::DeviceBuildInfo>& infos,
            const uint64_t* const pIndirectOffsets, const uint32_t* const pIndirectStrides,
            const uint32_t* const* const ppMaxPrimitiveCounts, const uint32_t totalGeometryCount
        ) override
        {
            // TODO: check for the raytracing feature enabled before wasting memory
            IGPUCommandPool::StackAllocation<VkAccelerationStructureGeometryMotionTrianglesDataNV> vk_vertexMotions(m_cmdpool,totalGeometryCount);
            if (!vk_vertexMotions)
                return false;

            return buildAccelerationStructuresIndirect_impl_impl<IGPUBottomLevelAccelerationStructure>(indirectRangeBuffer,infos,pIndirectOffsets,pIndirectStrides,ppMaxPrimitiveCounts,totalGeometryCount,vk_vertexMotions.data());
        }
        bool buildAccelerationStructuresIndirect_impl(
            const IGPUBuffer* indirectRangeBuffer, const core::SRange<const IGPUTopLevelAccelerationStructure::DeviceBuildInfo>& infos,
            const uint64_t* const pIndirectOffsets, const uint32_t* const pIndirectStrides, const uint32_t* const pMaxInstanceCounts
        ) override
        {
            const auto infoCount = infos.size();
            IGPUCommandPool::StackAllocation<const uint32_t*> vk_pMaxInstanceCounts(m_cmdpool,infoCount);
            if (!vk_pMaxInstanceCounts)
                return false;
            
            for (auto i=0u; i<infoCount; i++)
                vk_pMaxInstanceCounts[i] = pMaxInstanceCounts+i;
            return buildAccelerationStructuresIndirect_impl_impl<IGPUTopLevelAccelerationStructure>(indirectRangeBuffer,infos,pIndirectOffsets,pIndirectStrides,vk_pMaxInstanceCounts.data(),infoCount);
        }
        template<class AccelerationStructure> requires std::is_base_of_v<IGPUAccelerationStructure,AccelerationStructure>
        inline bool buildAccelerationStructuresIndirect_impl_impl(
            const IGPUBuffer* indirectRangeBuffer, const core::SRange<const typename AccelerationStructure::DeviceBuildInfo>& infos,
            const uint64_t* const pIndirectOffsets, const uint32_t* const pIndirectStrides,
            const uint32_t* const* const ppMaxPrimitiveOrInstanceCounts, const uint32_t totalGeometryCount,
            VkAccelerationStructureGeometryMotionTrianglesDataNV* out_vk_vertexMotions=nullptr
        )
        {
            const auto infoCount = infos.size();
            IGPUCommandPool::StackAllocation<VkDeviceAddress> indirectDeviceAddresses(m_cmdpool,infoCount);
            IGPUCommandPool::StackAllocation<VkAccelerationStructureBuildGeometryInfoKHR> vk_buildGeomsInfos(m_cmdpool,infoCount);
            IGPUCommandPool::StackAllocation<VkAccelerationStructureGeometryKHR> vk_geometries(m_cmdpool,totalGeometryCount);
            if (!indirectDeviceAddresses || !vk_geometries || !vk_buildGeomsInfos)
                return false;
            
            const auto baseIndirectAddress = indirectRangeBuffer->getDeviceAddress();
            auto out_vk_geoms = vk_geometries.data();
            for (auto i=0u; i<infoCount; i++)
            {
                getVkASBuildGeometryInfo<typename AccelerationStructure::DeviceBuildInfo>(infos[i],out_vk_geoms,out_vk_vertexMotions);
                indirectDeviceAddresses[i] = baseIndirectAddress+pIndirectOffsets[i];
            }
            getFunctionTable().vkCmdBuildAccelerationStructuresIndirectKHR(m_cmdbuf,infoCount,vk_buildGeomsInfos.data(),indirectDeviceAddresses.data(),pIndirectStrides,ppMaxPrimitiveOrInstanceCounts);
            return true;
        }

        bool copyAccelerationStructure_impl(const IGPUAccelerationStructure::CopyInfo& copyInfo) override;
        bool copyAccelerationStructureToMemory_impl(const IGPUAccelerationStructure::DeviceCopyToMemoryInfo& copyInfo) override;
        bool copyAccelerationStructureFromMemory_impl(const IGPUAccelerationStructure::DeviceCopyFromMemoryInfo& copyInfo) override;

        bool bindComputePipeline_impl(const IGPUComputePipeline* const pipeline) override;
        bool bindGraphicsPipeline_impl(const IGPUGraphicsPipeline* const pipeline) override;
        bool bindDescriptorSets_impl(const asset::E_PIPELINE_BIND_POINT pipelineBindPoint, const IGPUPipelineLayout* const layout, const uint32_t firstSet, const uint32_t descriptorSetCount, const IGPUDescriptorSet* const* const pDescriptorSets, const uint32_t dynamicOffsetCount = 0u, const uint32_t* const dynamicOffsets = nullptr) override;
        bool pushConstants_impl(const IGPUPipelineLayout* const layout, const core::bitflag<IGPUShader::E_SHADER_STAGE> stageFlags, const uint32_t offset, const uint32_t size, const void* const pValues) override;
        bool bindVertexBuffers_impl(const uint32_t firstBinding, const uint32_t bindingCount, const asset::SBufferBinding<const IGPUBuffer>* const pBindings) override;
        bool bindIndexBuffer_impl(const asset::SBufferBinding<const IGPUBuffer>& binding, const asset::E_INDEX_TYPE indexType) override;

        bool setScissor_impl(const uint32_t first, const uint32_t count, const VkRect2D* const pScissors) override;
        bool setViewport_impl(const uint32_t first, const uint32_t count, const asset::SViewport* const pViewports) override;

        bool resetQueryPool_impl(IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount) override;
        bool beginQuery_impl(IQueryPool* const queryPool, const uint32_t query, const core::bitflag<QUERY_CONTROL_FLAGS> flags = QUERY_CONTROL_FLAGS::NONE) override;
        bool endQuery_impl(IQueryPool* const queryPool, const uint32_t query) override;
        bool writeTimestamp_impl(const asset::PIPELINE_STAGE_FLAGS pipelineStage, IQueryPool* const queryPool, const uint32_t query) override;
        bool writeAccelerationStructureProperties_impl(const core::SRange<const IGPUAccelerationStructure*>& pAccelerationStructures, const IQueryPool::TYPE queryType, IQueryPool* const queryPool, const uint32_t firstQuery) override;
        bool copyQueryPoolResults_impl(const IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount, const asset::SBufferBinding<IGPUBuffer>& dstBuffer, const size_t stride, const core::bitflag<IQueryPool::RESULTS_FLAGS> flags) override;

        bool dispatch_impl(const uint32_t groupCountX, const uint32_t groupCountY, const uint32_t groupCountZ) override;
        bool dispatchIndirect_impl(const asset::SBufferBinding<const IGPUBuffer>& binding) override;

        bool beginRenderPass_impl(const SRenderpassBeginInfo& info, SUBPASS_CONTENTS contents) override;
        bool nextSubpass_impl(const SUBPASS_CONTENTS contents) override;
        bool endRenderPass_impl() override;

        bool clearAttachments_impl(const SClearAttachments& info) override;

        bool draw_impl(const uint32_t vertexCount, const uint32_t instanceCount, const uint32_t firstVertex, const uint32_t firstInstance) override;
        bool drawIndexed_impl(const uint32_t indexCount, const uint32_t instanceCount, const uint32_t firstIndex, const int32_t vertexOffset, const uint32_t firstInstance) override;
        bool drawIndirect_impl(const asset::SBufferBinding<const IGPUBuffer>& binding, const uint32_t drawCount, const uint32_t stride) override;
        bool drawIndexedIndirect_impl(const asset::SBufferBinding<const IGPUBuffer>& binding, const uint32_t drawCount, const uint32_t stride) override;
        bool drawIndirectCount_impl(const asset::SBufferBinding<const IGPUBuffer>& indirectBinding, const asset::SBufferBinding<const IGPUBuffer>& countBinding, const uint32_t maxDrawCount, const uint32_t stride) override;
        bool drawIndexedIndirectCount_impl(const asset::SBufferBinding<const IGPUBuffer>& indirectBinding, const asset::SBufferBinding<const IGPUBuffer>& countBinding, const uint32_t maxDrawCount, const uint32_t stride) override;

        bool blitImage_impl(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const SImageBlit* pRegions, const IGPUSampler::E_TEXTURE_FILTER filter) override;
        bool resolveImage_impl(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const SImageResolve* pRegions) override;

        bool executeCommands_impl(const uint32_t count, IGPUCommandBuffer* const* const cmdbufs) override;

    private:
        const VolkDeviceTable& getFunctionTable() const;

<<<<<<< HEAD
        VkCommandBuffer m_cmdbuf;
=======
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

    inline bool setDeviceMask_impl(uint32_t deviceMask) override final
    {
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
            getVkShaderStageFlagsFromShaderStage(stageFlags),
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

    bool insertDebugMarker(const char* name, const core::vector4df_SIMD& color) override final
    {
        // This is instance function loaded by volk (via vkGetInstanceProcAddr), so we have to check for validity of the function ptr
        if (vkCmdInsertDebugUtilsLabelEXT == 0)
            return false;

        VkDebugUtilsLabelEXT labelInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT };
        labelInfo.pLabelName = name;
        labelInfo.color[0] = color.x;
        labelInfo.color[1] = color.y;
        labelInfo.color[2] = color.z;
        labelInfo.color[3] = color.w;

        vkCmdBeginDebugUtilsLabelEXT(m_cmdbuf, &labelInfo);
        return true;
    }

    bool beginDebugMarker(const char* name, const core::vector4df_SIMD& color) override final
    {
        // This is instance function loaded by volk (via vkGetInstanceProcAddr), so we have to check for validity of the function ptr
        if (vkCmdBeginDebugUtilsLabelEXT == 0)
            return false;
        
        VkDebugUtilsLabelEXT labelInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT };
        labelInfo.pLabelName = name;
        labelInfo.color[0] = color.x;
        labelInfo.color[1] = color.y;
        labelInfo.color[2] = color.z;
        labelInfo.color[3] = color.w;
        vkCmdBeginDebugUtilsLabelEXT(m_cmdbuf, &labelInfo);

        return true;
    }

    bool endDebugMarker() override final
    {
        // This is instance function loaded by volk (via vkGetInstanceProcAddr), so we have to check for validity of the function ptr
        if (vkCmdEndDebugUtilsLabelEXT == 0)
            return false;
        vkCmdEndDebugUtilsLabelEXT(m_cmdbuf);
        return true;
    }

	inline const void* getNativeHandle() const override {return &m_cmdbuf;}
    VkCommandBuffer getInternalObject() const {return m_cmdbuf;}

private:
    VkCommandBuffer m_cmdbuf;
>>>>>>> 798939af864768c9d936d4810ae3718b8032f2c8
};  

}

#endif
