#ifndef _NBL_VIDEO_I_GPU_COMMAND_BUFFER_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_COMMAND_BUFFER_H_INCLUDED_

#include "nbl/asset/ICommandBuffer.h"

#include "nbl/video/IGPUEvent.h"
#include "nbl/video/IGPUDescriptorSet.h"
#include "nbl/video/IGPUComputePipeline.h"
#include "nbl/video/IGPUGraphicsPipeline.h"
#include "nbl/video/IGPUFramebuffer.h"
#include "nbl/video/IGPUCommandPool.h"

namespace nbl::video
{

class NBL_API2 IGPUCommandBuffer :
    public core::IReferenceCounted,
    public asset::ICommandBuffer<
        IGPUBuffer,
        IGPUImage,
        IGPUImageView,
        IGPURenderpass,
        IGPUFramebuffer,
        IGPUGraphicsPipeline,
        IGPUComputePipeline,
        IGPUDescriptorSet,
        IGPUPipelineLayout,
        IGPUEvent,
        IGPUCommandBuffer
    >,
    public IBackendObject
{
    using base_t = asset::ICommandBuffer<
        IGPUBuffer,
        IGPUImage,
        IGPUImageView,
        IGPURenderpass,
        IGPUFramebuffer,
        IGPUGraphicsPipeline,
        IGPUComputePipeline,
        IGPUDescriptorSet,
        IGPUPipelineLayout,
        IGPUEvent,
        IGPUCommandBuffer
    >;

public:
    inline bool isResettable() const
    {
        return m_cmdpool->getCreationFlags().hasFlags(IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
    }

    inline bool canReset() const
    {
        if(isResettable())
            return m_state != ES_PENDING;
        return false;
    }

    bool begin(core::bitflag<E_USAGE> flags, const SInheritanceInfo* inheritanceInfo = nullptr) override final;
    bool reset(core::bitflag<E_RESET_FLAGS> flags) override final;
    bool end() override final;

    bool bindIndexBuffer(const buffer_t* buffer, size_t offset, asset::E_INDEX_TYPE indexType) override final;
    bool drawIndirect(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride) override final;
    bool drawIndexedIndirect(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride) override final;
    bool drawIndirectCount(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride) override final;
    bool drawIndexedIndirectCount(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride) override final;
    bool beginRenderPass(const SRenderpassBeginInfo* pRenderPassBegin, asset::E_SUBPASS_CONTENTS content) override final;
    bool pipelineBarrier(core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> srcStageMask, core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> dstStageMask,
        core::bitflag<asset::E_DEPENDENCY_FLAGS> dependencyFlags,
        uint32_t memoryBarrierCount, const asset::SMemoryBarrier* pMemoryBarriers,
        uint32_t bufferMemoryBarrierCount, const SBufferMemoryBarrier* pBufferMemoryBarriers,
        uint32_t imageMemoryBarrierCount, const SImageMemoryBarrier* pImageMemoryBarriers) override final;
    bool bindDescriptorSets(asset::E_PIPELINE_BIND_POINT pipelineBindPoint, const pipeline_layout_t* layout, uint32_t firstSet, const uint32_t descriptorSetCount,
        const descriptor_set_t* const* const pDescriptorSets, const uint32_t dynamicOffsetCount = 0u, const uint32_t* dynamicOffsets = nullptr) override final;
    bool bindComputePipeline(const compute_pipeline_t* pipeline) final override;
    bool updateBuffer(buffer_t* dstBuffer, size_t dstOffset, size_t dataSize, const void* pData) override final;

    bool buildAccelerationStructures(const core::SRange<video::IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, video::IGPUAccelerationStructure::BuildRangeInfo* const* ppBuildRangeInfos) override final;
    bool buildAccelerationStructuresIndirect(const core::SRange<video::IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, const core::SRange<video::IGPUAccelerationStructure::DeviceAddressType>& pIndirectDeviceAddresses, const uint32_t* pIndirectStrides, const uint32_t* const* ppMaxPrimitiveCounts) override final;
    bool copyAccelerationStructure(const video::IGPUAccelerationStructure::CopyInfo& copyInfo) override final;
    bool copyAccelerationStructureToMemory(const video::IGPUAccelerationStructure::DeviceCopyToMemoryInfo& copyInfo) override final;
    bool copyAccelerationStructureFromMemory(const video::IGPUAccelerationStructure::DeviceCopyFromMemoryInfo& copyInfo) override final;

    bool resetQueryPool(video::IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount) override final;
    bool writeTimestamp(asset::E_PIPELINE_STAGE_FLAGS pipelineStage, video::IQueryPool* queryPool, uint32_t query) override final;

    bool writeAccelerationStructureProperties(const core::SRange<video::IGPUAccelerationStructure>& pAccelerationStructures, video::IQueryPool::E_QUERY_TYPE queryType, video::IQueryPool* queryPool, uint32_t firstQuery) override final;

    bool beginQuery(video::IQueryPool* queryPool, uint32_t query, core::bitflag<video::IQueryPool::E_QUERY_CONTROL_FLAGS> flags = video::IQueryPool::E_QUERY_CONTROL_FLAGS::EQCF_NONE) override final;
    bool endQuery(video::IQueryPool* queryPool, uint32_t query) override final;
    bool copyQueryPoolResults(video::IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, buffer_t* dstBuffer, size_t dstOffset, size_t stride, core::bitflag<video::IQueryPool::E_QUERY_RESULTS_FLAGS> flags) override final;
    bool setDeviceMask(uint32_t deviceMask) override final;
    bool bindGraphicsPipeline(const graphics_pipeline_t* pipeline) override final;
    bool pushConstants(const pipeline_layout_t* layout, core::bitflag<asset::IShader::E_SHADER_STAGE> stageFlags, uint32_t offset, uint32_t size, const void* pValues) override final;
    bool clearColorImage(image_t* image, asset::IImage::E_LAYOUT imageLayout, const asset::SClearColorValue* pColor, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges) override final;
    bool clearDepthStencilImage(image_t* image, asset::IImage::E_LAYOUT imageLayout, const asset::SClearDepthStencilValue* pDepthStencil, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges) override final;
    bool fillBuffer(buffer_t* dstBuffer, size_t dstOffset, size_t size, uint32_t data) override final;
    bool bindVertexBuffers(uint32_t firstBinding, uint32_t bindingCount, const buffer_t* const* const pBuffers, const size_t* pOffsets) override final;
    bool dispatchIndirect(const buffer_t* buffer, size_t offset) override final;
    bool setEvent(event_t* _event, const SDependencyInfo& depInfo) override final;
    bool resetEvent(event_t* _event, asset::E_PIPELINE_STAGE_FLAGS stageMask) override final;
    bool waitEvents(uint32_t eventCount, event_t* const* const pEvents, const SDependencyInfo* depInfos) override final;
    bool drawMeshBuffer(const meshbuffer_t* meshBuffer) override final;
    bool copyBuffer(const buffer_t* srcBuffer, buffer_t* dstBuffer, uint32_t regionCount, const asset::SBufferCopy* pRegions) override final;
    bool copyImage(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SImageCopy* pRegions) override final;
    bool copyBufferToImage(const buffer_t* srcBuffer, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) override final;
    bool blitImage(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageBlit* pRegions, asset::ISampler::E_TEXTURE_FILTER filter) final override;
    bool copyImageToBuffer(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, buffer_t* dstBuffer, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) override final;
    bool resolveImage(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageResolve* pRegions) override final;
    bool executeCommands(uint32_t count, cmdbuf_t* const* const cmdbufs) override final;

    inline uint32_t getQueueFamilyIndex() const { return m_cmdpool->getQueueFamilyIndex(); }
    inline IGPUCommandPool* getPool() const { return m_cmdpool.get(); }
    inline SInheritanceInfo getCachedInheritanceInfo() const { return m_cachedInheritanceInfo; }

    // OpenGL: nullptr, because commandbuffer doesn't exist in GL (we might expose the linked list command storage in the future)
    // Vulkan: const VkCommandBuffer*
    virtual const void* getNativeHandle() const = 0;

    inline const core::unordered_map<const IGPUDescriptorSet*, uint64_t>& getBoundDescriptorSetsRecord() const { return m_boundDescriptorSetsRecord; }

protected: 
    friend class IGPUQueue;

    IGPUCommandBuffer(core::smart_refctd_ptr<const ILogicalDevice>&& dev, E_LEVEL lvl, core::smart_refctd_ptr<IGPUCommandPool>&& _cmdpool, system::logger_opt_smart_ptr&& logger) : base_t(lvl), IBackendObject(std::move(dev)), m_cmdpool(_cmdpool), m_logger(std::move(logger))
    {
    }

    virtual ~IGPUCommandBuffer()
    {
        // Only release the resources if the parent pool has not been reset because if it has been then the resources will already be released.
        if (!checkForParentPoolReset())
        {
            releaseResourcesBackToPool();
        }
    }

    system::logger_opt_smart_ptr m_logger;
    core::smart_refctd_ptr<IGPUCommandPool> m_cmdpool;
    SInheritanceInfo m_cachedInheritanceInfo;

    inline bool validate_updateBuffer(IGPUBuffer* dstBuffer, size_t dstOffset, size_t dataSize, const void* pData)
    {
        if (!dstBuffer)
            return false;
        if (!this->isCompatibleDevicewise(dstBuffer))
            return false;
        if ((dstOffset & 0x03ull) != 0ull)
            return false;
        if ((dataSize & 0x03ull) != 0ull)
            return false;
        if (dataSize > 65536ull)
            return false;
        return dstBuffer->getCreationParams().usage.hasFlags(IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF);
    }

    static void bindDescriptorSets_generic(const IGPUPipelineLayout* _newLayout, uint32_t _first, uint32_t _count, const IGPUDescriptorSet* const* _descSets, const IGPUPipelineLayout** const _destPplnLayouts)
    {
        int32_t compatibilityLimits[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT]{};
        for (uint32_t i = 0u; i < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; i++)
        {
            const int32_t lim = _destPplnLayouts[i] ? //if no descriptor set bound at this index
                _destPplnLayouts[i]->isCompatibleUpToSet(IGPUPipelineLayout::DESCRIPTOR_SET_COUNT - 1u, _newLayout) : -1;

            compatibilityLimits[i] = lim;
        }

        /*
        https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/vkspec.html#descriptorsets-compatibility
        When binding a descriptor set (see Descriptor Set Binding) to set number N, if the previously bound descriptor sets for sets zero through N-1 were all bound using compatible pipeline layouts, then performing this binding does not disturb any of the lower numbered sets.
        */
        for (int32_t i = 0; i < static_cast<int32_t>(_first); ++i)
            if (compatibilityLimits[i] < i)
                _destPplnLayouts[i] = nullptr;
        /*
        If, additionally, the previous bound descriptor set for set N was bound using a pipeline layout compatible for set N, then the bindings in sets numbered greater than N are also not disturbed.
        */
        if (compatibilityLimits[_first] < static_cast<int32_t>(_first))
            for (uint32_t i = _first + 1u; i < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
                _destPplnLayouts[i] = nullptr;
    }

    virtual bool begin_impl(core::bitflag<E_USAGE> flags, const SInheritanceInfo* inheritanceInfo) = 0;
    virtual bool reset_impl(core::bitflag<E_RESET_FLAGS> flags) { return true; };
    virtual bool end_impl() = 0;

    virtual void releaseResourcesBackToPool_impl() {}
    virtual void checkForParentPoolReset_impl() const = 0;

    virtual void bindIndexBuffer_impl(const buffer_t* buffer, size_t offset, asset::E_INDEX_TYPE indexType) = 0;
    virtual bool drawIndirect_impl(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride) = 0;
    virtual bool drawIndexedIndirect_impl(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride) = 0;
    virtual bool drawIndirectCount_impl(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride) = 0;
    virtual bool drawIndexedIndirectCount_impl(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride) = 0;
    virtual bool beginRenderPass_impl(const SRenderpassBeginInfo* pRenderPassBegin, asset::E_SUBPASS_CONTENTS content) = 0;
    virtual bool pipelineBarrier_impl(core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> srcStageMask, core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> dstStageMask,
        core::bitflag<asset::E_DEPENDENCY_FLAGS> dependencyFlags,
        uint32_t memoryBarrierCount, const asset::SMemoryBarrier* pMemoryBarriers,
        uint32_t bufferMemoryBarrierCount, const SBufferMemoryBarrier* pBufferMemoryBarriers,
        uint32_t imageMemoryBarrierCount, const SImageMemoryBarrier* pImageMemoryBarriers) = 0;
    virtual bool bindDescriptorSets_impl(asset::E_PIPELINE_BIND_POINT pipelineBindPoint, const pipeline_layout_t* layout, uint32_t firstSet, const uint32_t descriptorSetCount,
        const descriptor_set_t* const* const pDescriptorSets, const uint32_t dynamicOffsetCount = 0u, const uint32_t* dynamicOffsets = nullptr) = 0;
    virtual void bindComputePipeline_impl(const compute_pipeline_t* pipeline) = 0;
    virtual bool updateBuffer_impl(buffer_t* dstBuffer, size_t dstOffset, size_t dataSize, const void* pData) = 0;

    virtual bool buildAccelerationStructures_impl(const core::SRange<video::IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, video::IGPUAccelerationStructure::BuildRangeInfo* const* ppBuildRangeInfos) { assert(!"Invalid code path."); return false; }
    virtual bool buildAccelerationStructuresIndirect_impl(const core::SRange<video::IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, const core::SRange<video::IGPUAccelerationStructure::DeviceAddressType>& pIndirectDeviceAddresses, const uint32_t* pIndirectStrides, const uint32_t* const* ppMaxPrimitiveCounts) { assert(!"Invalid code path."); return false; }
    virtual bool copyAccelerationStructure_impl(const video::IGPUAccelerationStructure::CopyInfo& copyInfo) { assert(!"Invalid code path."); return false; }
    virtual bool copyAccelerationStructureToMemory_impl(const video::IGPUAccelerationStructure::DeviceCopyToMemoryInfo& copyInfo) { assert(!"Invalid code path."); return false; }
    virtual bool copyAccelerationStructureFromMemory_impl(const video::IGPUAccelerationStructure::DeviceCopyFromMemoryInfo& copyInfo) { assert(!"Invaild code path."); return false; }

    virtual bool resetQueryPool_impl(video::IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount) = 0;
    virtual bool writeTimestamp_impl(asset::E_PIPELINE_STAGE_FLAGS pipelineStage, video::IQueryPool* queryPool, uint32_t query) = 0;
    virtual bool writeAccelerationStructureProperties_impl(const core::SRange<video::IGPUAccelerationStructure>& pAccelerationStructures, video::IQueryPool::E_QUERY_TYPE queryType, video::IQueryPool* queryPool, uint32_t firstQuery) { assert(!"Invalid code path."); return false; }
    virtual bool beginQuery_impl(video::IQueryPool* queryPool, uint32_t query, core::bitflag<video::IQueryPool::E_QUERY_CONTROL_FLAGS> flags = video::IQueryPool::E_QUERY_CONTROL_FLAGS::EQCF_NONE) = 0;
    virtual bool endQuery_impl(video::IQueryPool* queryPool, uint32_t query) = 0;
    virtual bool copyQueryPoolResults_impl(video::IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, buffer_t* dstBuffer, size_t dstOffset, size_t stride, core::bitflag<video::IQueryPool::E_QUERY_RESULTS_FLAGS> flags) = 0;
    virtual bool setDeviceMask_impl(uint32_t deviceMask) { assert(!"Invalid code path"); return false; };
    virtual bool bindGraphicsPipeline_impl(const graphics_pipeline_t* pipeline) = 0;
    virtual bool pushConstants_impl(const pipeline_layout_t* layout, core::bitflag<asset::IShader::E_SHADER_STAGE> stageFlags, uint32_t offset, uint32_t size, const void* pValues) = 0;
    virtual bool clearColorImage_impl(image_t* image, asset::IImage::E_LAYOUT imageLayout, const asset::SClearColorValue* pColor, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges) = 0;
    virtual bool clearDepthStencilImage_impl(image_t* image, asset::IImage::E_LAYOUT imageLayout, const asset::SClearDepthStencilValue* pDepthStencil, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges) = 0;
    virtual bool clearAttachments(uint32_t attachmentCount, const asset::SClearAttachment* pAttachments, uint32_t rectCount, const asset::SClearRect* pRects) = 0;
    virtual bool fillBuffer_impl(buffer_t* dstBuffer, size_t dstOffset, size_t size, uint32_t data) = 0;
    virtual void bindVertexBuffers_impl(uint32_t firstBinding, uint32_t bindingCount, const buffer_t* const* const pBuffers, const size_t* pOffsets) = 0;
    virtual bool dispatchIndirect_impl(const buffer_t* buffer, size_t offset) = 0;
    virtual bool setEvent_impl(event_t* _event, const SDependencyInfo& depInfo) = 0;
    virtual bool resetEvent_impl(event_t* _event, asset::E_PIPELINE_STAGE_FLAGS stageMask) = 0;
    virtual bool waitEvents_impl(uint32_t eventCount, event_t* const* const pEvents, const SDependencyInfo* depInfos) = 0;
    virtual bool copyBuffer_impl(const buffer_t* srcBuffer, buffer_t* dstBuffer, uint32_t regionCount, const asset::SBufferCopy* pRegions) = 0;
    virtual bool copyImage_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SImageCopy* pRegions) = 0;
    virtual bool copyBufferToImage_impl(const buffer_t* srcBuffer, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) = 0;
    virtual bool blitImage_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageBlit* pRegions, asset::ISampler::E_TEXTURE_FILTER filter) = 0;
    virtual bool copyImageToBuffer_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, buffer_t* dstBuffer, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) = 0;
    virtual bool resolveImage_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageResolve* pRegions) = 0;
    virtual bool executeCommands_impl(uint32_t count, cmdbuf_t* const* const cmdbufs) = 0;

private:
    // everything here is private on purpose so that derived class can't mess with these basic states
    inline bool checkForParentPoolReset()
    {
        if (m_cmdpool->getResetCounter() <= m_resetCheckedStamp)
            return false;

        m_resetCheckedStamp = m_cmdpool->getResetCounter();
        m_state = ES_INITIAL;

        m_boundDescriptorSetsRecord.clear();

        m_commandList.head = nullptr;
        m_commandList.tail = nullptr;

        checkForParentPoolReset_impl();

        return true;
    }

    inline void releaseResourcesBackToPool()
    {
        deleteCommandList();
        m_boundDescriptorSetsRecord.clear();
        releaseResourcesBackToPool_impl();
    }

    inline void deleteCommandList()
    {
        m_cmdpool->m_commandListPool.deleteList(m_commandList.head);
        m_commandList.head = nullptr;
        m_commandList.tail = nullptr;
    }

    inline bool checkStateBeforeRecording()
    {
        if (m_state != ES_RECORDING)
        {
            m_logger.log("Failed to record into command buffer: not in RECORDING state.", system::ILogger::ELL_ERROR);
            return false;
        }
        if (checkForParentPoolReset())
        {
            m_logger.log("Failed to record into command buffer: pool was reset since the recording begin() call.", system::ILogger::ELL_ERROR);
            return false;
        }
        return true;
    }

    uint64_t m_resetCheckedStamp = 0;

    // This bound descriptor set record doesn't include the descriptor sets whose layout has _any_ one of its bindings
    // created with IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT
    // or IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_UPDATE_UNUSED_WHILE_PENDING_BIT.
    core::unordered_map<const IGPUDescriptorSet*, uint64_t> m_boundDescriptorSetsRecord;
    
    IGPUCommandPool::CCommandSegmentListPool::SCommandSegmentList m_commandList;
};

}

#endif
