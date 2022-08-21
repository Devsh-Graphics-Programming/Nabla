#include "nbl/video/IGPUCommandBuffer.h"

namespace nbl::video
{

bool IGPUCommandBuffer::begin(core::bitflag<E_USAGE> flags, const SInheritanceInfo* inheritanceInfo)
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

        m_state = ES_INITIAL;
    }

    assert(m_state == ES_INITIAL);

    if (inheritanceInfo != nullptr)
        m_cachedInheritanceInfo = *inheritanceInfo;

    m_recordingFlags = flags;
    m_state = ES_RECORDING;

    return begin_impl(flags, inheritanceInfo);
}

bool IGPUCommandBuffer::reset(core::bitflag<E_RESET_FLAGS> flags)
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

bool IGPUCommandBuffer::end()
{
    if (m_state != ES_RECORDING)
    {
        m_logger.log("Failed to end command buffer: not in RECORDING state.", system::ILogger::ELL_ERROR);
        return false;
    }

    m_state = ES_EXECUTABLE;
    return end_impl();
}

bool IGPUCommandBuffer::bindIndexBuffer(const buffer_t* buffer, size_t offset, asset::E_INDEX_TYPE indexType)
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

bool IGPUCommandBuffer::drawIndirect(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride)
{
    if (!buffer || (buffer->getAPIType() != getAPIType()))
        return false;

    if (!m_cmdpool->emplace<IGPUCommandPool::CDrawIndirectCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IGPUBuffer>(buffer)))
        return false;

    drawIndirect_impl(buffer, offset, drawCount, stride);

    return true;
}

bool IGPUCommandBuffer::drawIndexedIndirect(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride)
{
    if (!buffer || buffer->getAPIType() != EAT_VULKAN)
        return false;

    if (!m_cmdpool->emplace<IGPUCommandPool::CDrawIndexedIndirectCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const buffer_t>(buffer)))
        return false;

    drawIndexedIndirect_impl(buffer, offset, drawCount, stride);

    return true;
}

bool IGPUCommandBuffer::drawIndirectCount(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride)
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

bool IGPUCommandBuffer::drawIndexedIndirectCount(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride)
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

bool IGPUCommandBuffer::beginRenderPass(const SRenderpassBeginInfo* pRenderPassBegin, asset::E_SUBPASS_CONTENTS content)
{
    const auto apiType = getAPIType();
    if ((apiType != pRenderPassBegin->renderpass->getAPIType()) || (apiType != pRenderPassBegin->framebuffer->getAPIType()))
        return false;

    if (!this->isCompatibleDevicewise(pRenderPassBegin->framebuffer.get()))
        return false;

    if (!m_cmdpool->emplace<IGPUCommandPool::CBeginRenderPassCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IGPURenderpass>(pRenderPassBegin->renderpass), core::smart_refctd_ptr<const IGPUFramebuffer>(pRenderPassBegin->framebuffer)))
        return false;

    return beginRenderPass_impl(pRenderPassBegin, content);
}

bool IGPUCommandBuffer::pipelineBarrier(core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> srcStageMask, core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> dstStageMask,
    core::bitflag<asset::E_DEPENDENCY_FLAGS> dependencyFlags,
    uint32_t memoryBarrierCount, const asset::SMemoryBarrier* pMemoryBarriers,
    uint32_t bufferMemoryBarrierCount, const SBufferMemoryBarrier* pBufferMemoryBarriers,
    uint32_t imageMemoryBarrierCount, const SImageMemoryBarrier* pImageMemoryBarriers)
{
    if ((memoryBarrierCount == 0u) && (bufferMemoryBarrierCount == 0u) && (imageMemoryBarrierCount == 0u))
        return false;

    constexpr auto MaxBarrierResourceCount = (1 << 12) / sizeof(void*);
    assert(bufferMemoryBarrierCount + imageMemoryBarrierCount <= MaxBarrierResourceCount);

    core::smart_refctd_ptr<const IGPUBuffer> bufferResources[MaxBarrierResourceCount];
    for (auto i = 0; i < bufferMemoryBarrierCount; ++i)
        bufferResources[i] = pBufferMemoryBarriers[i].buffer;

    core::smart_refctd_ptr<const IGPUImage> imageResources[MaxBarrierResourceCount];
    for (auto i = 0; i < imageMemoryBarrierCount; ++i)
        imageResources[i] = pImageMemoryBarriers[i].image;

    if (!m_cmdpool->emplace<IGPUCommandPool::CPipelineBarrierCmd>(m_segmentListHeadItr, m_segmentListTail, bufferMemoryBarrierCount, bufferResources, imageMemoryBarrierCount, imageResources))
        return false;

    pipelineBarrier_impl(srcStageMask, dstStageMask, dependencyFlags, memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);

    return true;
}

bool IGPUCommandBuffer::bindDescriptorSets(asset::E_PIPELINE_BIND_POINT pipelineBindPoint, const pipeline_layout_t* layout, uint32_t firstSet, uint32_t descriptorSetCount,
    const descriptor_set_t* const* const pDescriptorSets, const uint32_t dynamicOffsetCount, const uint32_t* dynamicOffsets)
{
    if (!this->isCompatibleDevicewise(layout))
        return false;

    if (layout->getAPIType() != getAPIType())
        return false;

    for (uint32_t i = 0u; i < descriptorSetCount; ++i)
    {
        if (pDescriptorSets[i])
        {
            if (!this->isCompatibleDevicewise(pDescriptorSets[i]))
            {
                m_logger.log("IGPUCommandBuffer::bindDescriptorSets failed, pDescriptorSets[%d] was not created by the same ILogicalDevice as the commandbuffer!", system::ILogger::ELL_ERROR, i);
                return false;
            }
            if (!pDescriptorSets[i]->getLayout()->isIdenticallyDefined(layout->getDescriptorSetLayout(firstSet + i)))
            {
                m_logger.log("IGPUCommandBuffer::bindDescriptorSets failed, pDescriptorSets[%d] not identically defined as layout's %dth descriptor layout!", system::ILogger::ELL_ERROR, i, firstSet + i);
                return false;
            }
        }
    }

    core::smart_refctd_ptr<const video::IGPUDescriptorSet> descriptorSets_refctd[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT] = { nullptr };
    for (auto i = 0; i < descriptorSetCount; ++i)
    {
        assert(i < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT);
        descriptorSets_refctd[i] = core::smart_refctd_ptr<const video::IGPUDescriptorSet>(pDescriptorSets[i]);
    }

    if (!m_cmdpool->emplace<IGPUCommandPool::CBindDescriptorSetsCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IGPUPipelineLayout>(layout), descriptorSetCount, descriptorSets_refctd))
        return false;

    return bindDescriptorSets_impl(pipelineBindPoint, layout, firstSet, descriptorSetCount, pDescriptorSets, dynamicOffsetCount, dynamicOffsets);
}

bool IGPUCommandBuffer::bindComputePipeline(const compute_pipeline_t* pipeline)
{
    if (!this->isCompatibleDevicewise(pipeline))
        return false;

    if (pipeline->getAPIType() != getAPIType())
        return false;

    if (!m_cmdpool->emplace<IGPUCommandPool::CBindComputePipelineCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IGPUComputePipeline>(pipeline)))
        return false;

    bindComputePipeline_impl(pipeline);

    return true;
}

}