#ifndef __NBL_I_GPU_COMMAND_BUFFER_H_INCLUDED__
#define __NBL_I_GPU_COMMAND_BUFFER_H_INCLUDED__

#include "nbl/asset/ICommandBuffer.h"
/*
#include "nbl/video/IGPUImage.h"
#include "nbl/video/IGPUImageView.h"
#include "nbl/video/IGPURenderpass.h"
#include "nbl/video/IGPUFramebuffer.h"
#include "nbl/video/IGPUGraphicsPipeline.h"
*/
#include "nbl/video/IGPUDescriptorSet.h"
/*
#include "nbl/video/IGPUPipelineLayout.h"
*/
#include "nbl/video/IGPUEvent.h"
#include "nbl/video/IGPUComputePipeline.h"
#include "nbl/video/IGPUFramebuffer.h"
#include "nbl/video/IGPUGraphicsPipeline.h"
#include "nbl/video/IGPUCommandPool.h"

namespace nbl::video
{

class NBL_API IGPUCommandBuffer :
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
    bool resetQueryPool(video::IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount) override final;
    bool writeTimestamp(asset::E_PIPELINE_STAGE_FLAGS pipelineStage, video::IQueryPool* queryPool, uint32_t query) override final;
    bool beginQuery(video::IQueryPool* queryPool, uint32_t query, core::bitflag<video::IQueryPool::E_QUERY_CONTROL_FLAGS> flags = video::IQueryPool::E_QUERY_CONTROL_FLAGS::EQCF_NONE) override final;
    bool endQuery(video::IQueryPool* queryPool, uint32_t query) override final;
    bool copyQueryPoolResults(video::IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, buffer_t* dstBuffer, size_t dstOffset, size_t stride, core::bitflag<video::IQueryPool::E_QUERY_RESULTS_FLAGS> flags) override final;
    bool bindGraphicsPipeline(const graphics_pipeline_t* pipeline) override final;
    bool pushConstants(const pipeline_layout_t* layout, core::bitflag<asset::IShader::E_SHADER_STAGE> stageFlags, uint32_t offset, uint32_t size, const void* pValues) override final;
    bool bindVertexBuffers(uint32_t firstBinding, uint32_t bindingCount, const buffer_t* const* const pBuffers, const size_t* pOffsets) override final;
    bool dispatchIndirect(const buffer_t* buffer, size_t offset) override final;
    bool waitEvents(uint32_t eventCount, event_t* const* const pEvents, const SDependencyInfo* depInfos) override final;
    bool drawMeshBuffer(const meshbuffer_t* meshBuffer) override final;
    bool copyBuffer(const buffer_t* srcBuffer, buffer_t* dstBuffer, uint32_t regionCount, const asset::SBufferCopy* pRegions) override final;
    bool copyImage(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SImageCopy* pRegions) override final;
    bool copyBufferToImage(const buffer_t* srcBuffer, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) override final;
    bool blitImage(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageBlit* pRegions, asset::ISampler::E_TEXTURE_FILTER filter) final override;
    bool copyImageToBuffer(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, buffer_t* dstBuffer, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) override final;
    bool executeCommands(uint32_t count, cmdbuf_t* const* const cmdbufs) override final;

    inline uint32_t getQueueFamilyIndex() const { return m_cmdpool->getQueueFamilyIndex(); }

    inline IGPUCommandPool* getPool() const { return m_cmdpool.get(); }

    bool regenerateMipmaps(IGPUImage* img, uint32_t lastReadyMip, asset::IImage::E_ASPECT_FLAGS aspect) override
    {
        const uint32_t qfam = getQueueFamilyIndex();

        IGPUCommandBuffer::SImageMemoryBarrier barrier = {};
        barrier.srcQueueFamilyIndex = qfam;
        barrier.dstQueueFamilyIndex = qfam;
        barrier.image = core::smart_refctd_ptr<video::IGPUImage>(img);
        barrier.subresourceRange.aspectMask = aspect;
        barrier.subresourceRange.levelCount = 1u;
        barrier.subresourceRange.baseArrayLayer = 0u;
        barrier.subresourceRange.layerCount = img->getCreationParameters().arrayLayers;

        asset::SImageBlit blitRegion = {};
        blitRegion.srcSubresource.aspectMask = barrier.subresourceRange.aspectMask;
        blitRegion.srcSubresource.baseArrayLayer = barrier.subresourceRange.baseArrayLayer;
        blitRegion.srcSubresource.layerCount = barrier.subresourceRange.layerCount;
        blitRegion.srcOffsets[0] = { 0, 0, 0 };

        blitRegion.dstSubresource.aspectMask = barrier.subresourceRange.aspectMask;
        blitRegion.dstSubresource.baseArrayLayer = barrier.subresourceRange.baseArrayLayer;
        blitRegion.dstSubresource.layerCount = barrier.subresourceRange.layerCount;
        blitRegion.dstOffsets[0] = { 0, 0, 0 };

        auto mipsize = img->getMipSize(lastReadyMip);

        uint32_t mipWidth = mipsize.x;
        uint32_t mipHeight = mipsize.y;
        uint32_t mipDepth = mipsize.z;
        for (uint32_t i = lastReadyMip + 1u; i < img->getCreationParameters().mipLevels; ++i)
        {
            const uint32_t srcLoD = i - 1u;
            const uint32_t dstLoD = i;

            barrier.barrier.srcAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
            barrier.barrier.dstAccessMask = asset::EAF_TRANSFER_READ_BIT;
            barrier.oldLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = asset::IImage::EL_TRANSFER_SRC_OPTIMAL;
            barrier.subresourceRange.baseMipLevel = dstLoD;

            if (srcLoD > lastReadyMip)
            {
                if (!pipelineBarrier(asset::EPSF_TRANSFER_BIT, asset::EPSF_TRANSFER_BIT, static_cast<asset::E_DEPENDENCY_FLAGS>(0u), 0u, nullptr, 0u, nullptr, 1u, &barrier))
                    return false;
            }

            const auto srcMipSz = img->getMipSize(srcLoD);

            blitRegion.srcSubresource.mipLevel = srcLoD;
            blitRegion.srcOffsets[1] = { srcMipSz.x, srcMipSz.y, srcMipSz.z };

            blitRegion.dstSubresource.mipLevel = dstLoD;
            blitRegion.dstOffsets[1] = { mipWidth, mipHeight, mipDepth };

            if (!blitImage(img, asset::IImage::EL_TRANSFER_SRC_OPTIMAL, img, asset::IImage::EL_TRANSFER_DST_OPTIMAL, 1u, &blitRegion, asset::ISampler::ETF_LINEAR))
                return false;

            if (mipWidth > 1u) mipWidth /= 2u;
            if (mipHeight > 1u) mipHeight /= 2u;
            if (mipDepth > 1u) mipDepth /= 2u;
        }

        return true;
    }

    SInheritanceInfo getCachedInheritanceInfo() const
    {
        return m_cachedInheritanceInfo;
    }

    // OpenGL: nullptr, because commandbuffer doesn't exist in GL (we might expose the linked list command storage in the future)
    // Vulkan: const VkCommandBuffer*
    virtual const void* getNativeHandle() const = 0;

protected: 
    friend class IGPUQueue;

    IGPUCommandBuffer(core::smart_refctd_ptr<const ILogicalDevice>&& dev, E_LEVEL lvl, core::smart_refctd_ptr<IGPUCommandPool>&& _cmdpool, system::logger_opt_smart_ptr&& logger) : base_t(lvl), IBackendObject(std::move(dev)), m_cmdpool(_cmdpool), m_logger(std::move(logger))
    {
    }

    virtual ~IGPUCommandBuffer()
    {
        if (!checkForParentPoolReset())
            releaseResourcesBackToPool();
    }

    system::logger_opt_smart_ptr m_logger;
    core::smart_refctd_ptr<IGPUCommandPool> m_cmdpool;
    SInheritanceInfo m_cachedInheritanceInfo;

    inline bool validate_updateBuffer(IGPUBuffer* dstBuffer, size_t dstOffset, size_t dataSize, const void* pData)
    {
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
    virtual bool resetQueryPool_impl(video::IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount) = 0;
    virtual bool writeTimestamp_impl(asset::E_PIPELINE_STAGE_FLAGS pipelineStage, video::IQueryPool* queryPool, uint32_t query) = 0;
    virtual bool beginQuery_impl(video::IQueryPool* queryPool, uint32_t query, core::bitflag<video::IQueryPool::E_QUERY_CONTROL_FLAGS> flags = video::IQueryPool::E_QUERY_CONTROL_FLAGS::EQCF_NONE) = 0;
    virtual bool endQuery_impl(video::IQueryPool* queryPool, uint32_t query) = 0;
    virtual bool copyQueryPoolResults_impl(video::IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, buffer_t* dstBuffer, size_t dstOffset, size_t stride, core::bitflag<video::IQueryPool::E_QUERY_RESULTS_FLAGS> flags) = 0;
    virtual bool bindGraphicsPipeline_impl(const graphics_pipeline_t* pipeline) = 0;
    virtual bool pushConstants_impl(const pipeline_layout_t* layout, core::bitflag<asset::IShader::E_SHADER_STAGE> stageFlags, uint32_t offset, uint32_t size, const void* pValues) = 0;
    virtual void bindVertexBuffers_impl(uint32_t firstBinding, uint32_t bindingCount, const buffer_t* const* const pBuffers, const size_t* pOffsets) = 0;
    virtual bool dispatchIndirect_impl(const buffer_t* buffer, size_t offset) = 0;
    virtual bool waitEvents_impl(uint32_t eventCount, event_t* const* const pEvents, const SDependencyInfo* depInfos) = 0;
    virtual bool copyBuffer_impl(const buffer_t* srcBuffer, buffer_t* dstBuffer, uint32_t regionCount, const asset::SBufferCopy* pRegions) = 0;
    virtual bool copyImage_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SImageCopy* pRegions) = 0;
    virtual bool copyBufferToImage_impl(const buffer_t* srcBuffer, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) = 0;
    virtual bool blitImage_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageBlit* pRegions, asset::ISampler::E_TEXTURE_FILTER filter) = 0;
    virtual bool copyImageToBuffer_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, buffer_t* dstBuffer, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) = 0;
    virtual bool executeCommands_impl(uint32_t count, cmdbuf_t* const* const cmdbufs) = 0;

private:
    // Be wary of making it protected/calling it in the derived classes because it sets state which will overwrite the state set in base class methods.
    inline bool checkForParentPoolReset()
    {
        if (m_cmdpool->getResetCounter() <= m_resetCheckedStamp)
            return false;

        m_resetCheckedStamp = m_cmdpool->getResetCounter();
        m_state = ES_INITIAL;

        m_segmentListHeadItr.m_cmd = nullptr;
        m_segmentListHeadItr.m_segment = nullptr;
        m_segmentListTail = nullptr;

        return true;
    }

    inline void releaseResourcesBackToPool()
    {
        m_cmdpool->deleteCommandSegmentList(m_segmentListHeadItr, m_segmentListTail);
        releaseResourcesBackToPool_impl();
    }

    uint32_t m_resetCheckedStamp = 0;
    
    IGPUCommandPool::CCommandSegment::Iterator m_segmentListHeadItr = {};
    IGPUCommandPool::CCommandSegment* m_segmentListTail = nullptr;
};

}

#endif
