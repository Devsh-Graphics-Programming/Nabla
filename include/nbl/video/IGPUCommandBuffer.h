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

    bool begin(core::bitflag<E_USAGE> flags, const SInheritanceInfo* inheritanceInfo = nullptr) override final
    {
        if (m_state == ES_RECORDING || m_state == ES_PENDING)
        {
            m_logger.log("Failed to begin command buffer: command buffer must not be in RECORDING or PENDING state.", system::ILogger::ELL_ERROR);
            return false;
        }

        if (m_level == EL_PRIMARY && (flags.hasFlags(static_cast<E_USAGE>(EU_ONE_TIME_SUBMIT_BIT | EU_SIMULTANEOUS_USE_BIT))))
        {
            m_logger.log("Failed to begin command buffer: a primary command buffer must not have both EU_ONE_TIME_SUBMIT_BIT and EU_SIMULTANEOUS_USE_BIT set.", system::ILogger::ELL_ERROR);
            return false;
        }

        if (m_level == EL_SECONDARY && inheritanceInfo == nullptr)
        {
            m_logger.log("Failed to begin command buffer: a secondary command buffer must have inheritance info.", system::ILogger::ELL_ERROR);
            return false;
        }

        checkForParentPoolReset();

        if (m_state != ES_INITIAL)
        {
            releaseResourcesBackToPool();

            if (!canReset())
            {
                m_logger.log("Failed to begin command buffer: command buffer allocated from a command pool with ECF_RESET_COMMAND_BUFFER_BIT flag not set cannot be reset, and command buffer not in INITIAL state.", system::ILogger::ELL_ERROR);
                m_state = ES_INVALID;
                return false;
            }
        }

        assert(m_state == ES_INITIAL);

        if (inheritanceInfo != nullptr)
            m_cachedInheritanceInfo = *inheritanceInfo;

        m_recordingFlags = flags;
        m_state = ES_RECORDING;

        return begin_impl(flags, inheritanceInfo);
    }

    bool reset(core::bitflag<E_RESET_FLAGS> flags) override final
    {
        if (!canReset())
        {
            m_logger.log("Failed to reset command buffer.", system::ILogger::ELL_ERROR);
            m_state = ES_INVALID;
            return false;
        }

        if (checkForParentPoolReset())
            return true;

        releaseResourcesBackToPool();
        m_state = ES_INITIAL;
        
        return reset_impl(flags);
    }

    bool end() override final
    {
        if (m_state != ES_RECORDING)
        {
            m_logger.log("Failed to end command buffer: not in RECORDING state.", system::ILogger::ELL_ERROR);
            return false;
        }

        m_state = ES_EXECUTABLE;
        return end_impl();
    }

    bool bindIndexBuffer(const buffer_t* buffer, size_t offset, asset::E_INDEX_TYPE indexType) override final
    {
        if (!buffer || (buffer->getAPIType() != getAPIType()))
            return false;

        if (!this->isCompatibleDevicewise(buffer))
            return false;

        if (!m_cmdpool->emplace<IGPUCommandPool::CBindIndexBufferCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IGPUBuffer>(buffer)))
            return false;

        bindIndexBuffer_impl(buffer, offset, indexType);

        return true;
    }

    bool drawIndirect(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride) override final
    {
        if (!buffer || (buffer->getAPIType() != getAPIType()))
            return false;

        if (!m_cmdpool->emplace<IGPUCommandPool::CDrawIndirectCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IGPUBuffer>(buffer)))
            return false;

        drawIndirect_impl(buffer, offset, drawCount, stride);

        return true;
    }

    bool drawIndexedIndirect(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride) override final
    {
        if (!buffer || buffer->getAPIType() != EAT_VULKAN)
            return false;

        if (!m_cmdpool->emplace<IGPUCommandPool::CDrawIndexedIndirectCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const buffer_t>(buffer)))
            return false;

        drawIndexedIndirect_impl(buffer, offset, drawCount, stride);

        return true;
    }

    bool drawIndirectCount(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride) override final
    {
        if (!buffer || buffer->getAPIType() != EAT_VULKAN)
            return false;

        if (!countBuffer || countBuffer->getAPIType() != EAT_VULKAN)
            return false;

        if (!m_cmdpool->emplace<IGPUCommandPool::CDrawIndirectCountCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const buffer_t>(buffer), core::smart_refctd_ptr<const buffer_t>(countBuffer)))
            return false;

        drawIndirectCount_impl(buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);

        return true;
    }

    bool drawIndexedIndirectCount(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride) override final
    {
        if (!buffer || buffer->getAPIType() != EAT_VULKAN)
            return false;

        if (!countBuffer || countBuffer->getAPIType() != EAT_VULKAN)
            return false;

        if (!m_cmdpool->emplace<IGPUCommandPool::CDrawIndexedIndirectCountCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const buffer_t>(buffer), core::smart_refctd_ptr<const buffer_t>(countBuffer)))
            return false;

        drawIndexedIndirectCount_impl(buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);

        return true;
    }

    bool beginRenderPass(const SRenderpassBeginInfo* pRenderPassBegin, asset::E_SUBPASS_CONTENTS content)
    {
        const auto apiType = getAPIType();
        if ((apiType != pRenderPassBegin->renderpass->getAPIType()) || (apiType != pRenderPassBegin->framebuffer->getAPIType()))
            return false;

        if (!m_cmdpool->emplace<IGPUCommandPool::CBeginRenderPassCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IGPURenderpass>(pRenderPassBegin->renderpass), core::smart_refctd_ptr<const IGPUFramebuffer>(pRenderPassBegin->framebuffer)))
            return false;

        return beginRenderPass_impl(pRenderPassBegin, content);
    }

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
            barrier.oldLayout = asset::EIL_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = asset::EIL_TRANSFER_SRC_OPTIMAL;
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

            if (!blitImage(img, asset::EIL_TRANSFER_SRC_OPTIMAL, img, asset::EIL_TRANSFER_DST_OPTIMAL, 1u, &blitRegion, asset::ISampler::ETF_LINEAR))
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
        return dstBuffer->getCachedCreationParams().canUpdateSubRange;
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
    virtual void drawIndirect_impl(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride) = 0;
    virtual void drawIndexedIndirect_impl(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride) = 0;
    virtual void drawIndirectCount_impl(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride) = 0;
    virtual void drawIndexedIndirectCount_impl(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride) = 0;

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
