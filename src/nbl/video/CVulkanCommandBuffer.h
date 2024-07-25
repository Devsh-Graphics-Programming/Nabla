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
        bool waitEvents_impl(const std::span<IEvent*> events, const SEventDependencyInfo* depInfos) override;
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
            const std::span<const IGPUBottomLevelAccelerationStructure::DeviceBuildInfo> infos,
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
        inline bool buildAccelerationStructures_impl(const std::span<const IGPUTopLevelAccelerationStructure::DeviceBuildInfo> infos, const IGPUTopLevelAccelerationStructure::BuildRangeInfo* const pBuildRangeInfos) override
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
            const std::span<const typename AccelerationStructure::DeviceBuildInfo> infos,
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
            const IGPUBuffer* indirectRangeBuffer, const std::span<const IGPUBottomLevelAccelerationStructure::DeviceBuildInfo> infos,
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
            const IGPUBuffer* indirectRangeBuffer, const std::span<const IGPUTopLevelAccelerationStructure::DeviceBuildInfo> infos,
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
            const IGPUBuffer* indirectRangeBuffer, const std::span<const typename AccelerationStructure::DeviceBuildInfo> infos,
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
        bool setLineWidth_impl(const float width) override;
        bool setDepthBias_impl(const float depthBiasConstantFactor, const float depthBiasClamp, const float depthBiasSlopeFactor) override;
        bool setBlendConstants_impl(const hlsl::float32_t4& constants) override;
        bool setDepthBounds_impl(const float minDepthBounds, const float maxDepthBounds) override;
        bool setStencilCompareMask_impl(const asset::E_FACE_CULL_MODE faces, const uint8_t compareMask) override;
        bool setStencilWriteMask_impl(const asset::E_FACE_CULL_MODE faces, const uint8_t writeMask) override;
        bool setStencilReference_impl(const asset::E_FACE_CULL_MODE faces, const uint8_t reference) override;

        bool resetQueryPool_impl(IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount) override;
        bool beginQuery_impl(IQueryPool* const queryPool, const uint32_t query, const core::bitflag<QUERY_CONTROL_FLAGS> flags = QUERY_CONTROL_FLAGS::NONE) override;
        bool endQuery_impl(IQueryPool* const queryPool, const uint32_t query) override;
        bool writeTimestamp_impl(const asset::PIPELINE_STAGE_FLAGS pipelineStage, IQueryPool* const queryPool, const uint32_t query) override;
        bool writeAccelerationStructureProperties_impl(const std::span<const IGPUAccelerationStructure* const> pAccelerationStructures, const IQueryPool::TYPE queryType, IQueryPool* const queryPool, const uint32_t firstQuery) override;
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

        bool blitImage_impl(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const std::span<const SImageBlit> regions, const IGPUSampler::E_TEXTURE_FILTER filter) override;
        bool resolveImage_impl(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const SImageResolve* pRegions) override;

        bool executeCommands_impl(const uint32_t count, IGPUCommandBuffer* const* const cmdbufs) override;

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

    private:
        virtual ~CVulkanCommandBuffer() final
        {
            out_of_order_dtor();
        }

        const VolkDeviceTable& getFunctionTable() const;

        VkCommandBuffer m_cmdbuf;
};  

}

#endif
