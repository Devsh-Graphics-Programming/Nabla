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

    if (drawCount == 0u)
        return false;

    if (!m_cmdpool->emplace<IGPUCommandPool::CDrawIndirectCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IGPUBuffer>(buffer)))
        return false;

    return drawIndirect_impl(buffer, offset, drawCount, stride);
}

bool IGPUCommandBuffer::drawIndexedIndirect(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride)
{
    if (!buffer || buffer->getAPIType() != getAPIType())
        return false;

    if (drawCount == 0u)
        return false;

    if (!m_cmdpool->emplace<IGPUCommandPool::CDrawIndexedIndirectCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const buffer_t>(buffer)))
        return false;

    drawIndexedIndirect_impl(buffer, offset, drawCount, stride);

    return true;
}

bool IGPUCommandBuffer::drawIndirectCount(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride)
{
    if (!buffer || buffer->getAPIType() != getAPIType())
        return false;

    if (!countBuffer || countBuffer->getAPIType() != getAPIType())
        return false;

    if (maxDrawCount == 0u)
        return false;

    if (!m_cmdpool->emplace<IGPUCommandPool::CDrawIndirectCountCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const buffer_t>(buffer), core::smart_refctd_ptr<const buffer_t>(countBuffer)))
        return false;

    return drawIndirectCount_impl(buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
}

bool IGPUCommandBuffer::drawIndexedIndirectCount(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride)
{
    if (!buffer || buffer->getAPIType() != getAPIType())
        return false;

    if (!countBuffer || countBuffer->getAPIType() != getAPIType())
        return false;

    if (maxDrawCount == 0u)
        return false;

    if (!m_cmdpool->emplace<IGPUCommandPool::CDrawIndexedIndirectCountCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const buffer_t>(buffer), core::smart_refctd_ptr<const buffer_t>(countBuffer)))
        return false;

    return drawIndexedIndirectCount_impl(buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);}

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

bool IGPUCommandBuffer::bindDescriptorSets(asset::E_PIPELINE_BIND_POINT pipelineBindPoint, const pipeline_layout_t* layout, uint32_t firstSet, const uint32_t descriptorSetCount,
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

bool IGPUCommandBuffer::updateBuffer(buffer_t* dstBuffer, size_t dstOffset, size_t dataSize, const void* pData)
{
    if (!dstBuffer || dstBuffer->getAPIType() != getAPIType())
        return false;

    if (!validate_updateBuffer(dstBuffer, dstOffset, dataSize, pData))
        return false;

    if (!m_cmdpool->emplace<IGPUCommandPool::CUpdateBufferCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IGPUBuffer>(dstBuffer)))
        return false;

    return updateBuffer_impl(dstBuffer, dstOffset, dataSize, pData);
}

bool IGPUCommandBuffer::resetQueryPool(video::IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount)
{
    if (!queryPool || !this->isCompatibleDevicewise(queryPool))
        return false;

    if (!m_cmdpool->emplace<IGPUCommandPool::CResetQueryPoolCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IQueryPool>(queryPool)))
        return false;

    return resetQueryPool_impl(queryPool, firstQuery, queryCount);
}

bool IGPUCommandBuffer::writeTimestamp(asset::E_PIPELINE_STAGE_FLAGS pipelineStage, video::IQueryPool* queryPool, uint32_t query)
{
    if (!queryPool || !this->isCompatibleDevicewise(queryPool))
        return false;

    assert(core::isPoT(static_cast<uint32_t>(pipelineStage))); // should only be 1 stage (1 bit set)

    if (!m_cmdpool->emplace<IGPUCommandPool::CWriteTimestampCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IQueryPool>(queryPool)))
        return false;

    return writeTimestamp_impl(pipelineStage, queryPool, query);
}

bool IGPUCommandBuffer::beginQuery(video::IQueryPool* queryPool, uint32_t query, core::bitflag<video::IQueryPool::E_QUERY_CONTROL_FLAGS> flags)
{
    if (!queryPool || !this->isCompatibleDevicewise(queryPool))
        return false;

    if (!m_cmdpool->emplace<IGPUCommandPool::CBeginQueryCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IQueryPool>(queryPool)))
        return false;

    return beginQuery_impl(queryPool, query, flags);
}

bool IGPUCommandBuffer::endQuery(video::IQueryPool* queryPool, uint32_t query)
{
    if (!queryPool || !this->isCompatibleDevicewise(queryPool))
        return false;

    if (!m_cmdpool->emplace<IGPUCommandPool::CEndQueryCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IQueryPool>(queryPool)))
        return false;

    return endQuery_impl(queryPool, query);
}

bool IGPUCommandBuffer::copyQueryPoolResults(video::IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, buffer_t* dstBuffer, size_t dstOffset, size_t stride, core::bitflag<video::IQueryPool::E_QUERY_RESULTS_FLAGS> flags)
{
    if (!queryPool || !this->isCompatibleDevicewise(queryPool))
        return false;

    if (!dstBuffer || !this->isCompatibleDevicewise(dstBuffer))
        return false;

    if (!m_cmdpool->emplace<IGPUCommandPool::CCopyQueryPoolResultsCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IQueryPool>(queryPool), core::smart_refctd_ptr<const IGPUBuffer>(dstBuffer)))
        return false;

    return copyQueryPoolResults_impl(queryPool, firstQuery, queryCount, dstBuffer, dstOffset, stride, flags);
}

bool IGPUCommandBuffer::bindGraphicsPipeline(const graphics_pipeline_t* pipeline)
{
    if (!pipeline || pipeline->getAPIType() != getAPIType())
        return false;

    if (!this->isCompatibleDevicewise(pipeline))
        return false;

    if (!m_cmdpool->emplace<IGPUCommandPool::CBindGraphicsPipelineCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IGPUGraphicsPipeline>(pipeline)))
        return false;

    return bindGraphicsPipeline_impl(pipeline);
}

bool IGPUCommandBuffer::pushConstants(const pipeline_layout_t* layout, core::bitflag<asset::IShader::E_SHADER_STAGE> stageFlags, uint32_t offset, uint32_t size, const void* pValues)
{
    if (layout->getAPIType() != getAPIType())
        return false;

    if (!m_cmdpool->emplace<IGPUCommandPool::CPushConstantsCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IGPUPipelineLayout>(layout)))
        return false;

    pushConstants_impl(layout, stageFlags, offset, size, pValues);

    return true;
}

bool IGPUCommandBuffer::clearColorImage(image_t* image, asset::IImage::E_LAYOUT imageLayout, const asset::SClearColorValue* pColor, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges)
{
    if (!image || image->getAPIType() != getAPIType())
        return false;

    if (!this->isCompatibleDevicewise(image))
        return false;

    if (!m_cmdpool->emplace<IGPUCommandPool::CClearColorImageCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IGPUImage>(image)))
        return false;

    return clearColorImage_impl(image, imageLayout, pColor, rangeCount, pRanges);
}

bool IGPUCommandBuffer::clearDepthStencilImage(image_t* image, asset::IImage::E_LAYOUT imageLayout, const asset::SClearDepthStencilValue* pDepthStencil, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges)
{
    if (!image || image->getAPIType() != getAPIType())
        return false;

    if (!this->isCompatibleDevicewise(image))
        return false;

    if (!m_cmdpool->emplace<IGPUCommandPool::CClearDepthStencilImageCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IGPUImage>(image)))
        return false;

    return clearDepthStencilImage_impl(image, imageLayout, pDepthStencil, rangeCount, pRanges);
}

bool IGPUCommandBuffer::fillBuffer(buffer_t* dstBuffer, size_t dstOffset, size_t size, uint32_t data)
{
    if (!dstBuffer || dstBuffer->getAPIType() != getAPIType())
        return false;

    if (!this->isCompatibleDevicewise(dstBuffer))
        return false;

    if (!m_cmdpool->emplace<IGPUCommandPool::CFillBufferCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IGPUBuffer>(dstBuffer)))
        return false;

    return fillBuffer_impl(dstBuffer, dstOffset, size, data);
}

bool IGPUCommandBuffer::bindVertexBuffers(uint32_t firstBinding, uint32_t bindingCount, const buffer_t* const* const pBuffers, const size_t* pOffsets)
{
    for (uint32_t i = 0u; i < bindingCount; ++i)
    {
        if (pBuffers[i] && !this->isCompatibleDevicewise(pBuffers[i]))
            return false;
    }

    if (!m_cmdpool->emplace<IGPUCommandPool::CBindVertexBuffersCmd>(m_segmentListHeadItr, m_segmentListTail, firstBinding, bindingCount, pBuffers))
        return false;

    bindVertexBuffers_impl(firstBinding, bindingCount, pBuffers, pOffsets);

    return true;
}

bool IGPUCommandBuffer::dispatchIndirect(const buffer_t* buffer, size_t offset)
{
    if (!buffer || buffer->getAPIType() != getAPIType())
        return false;

    if (!this->isCompatibleDevicewise(buffer))
        return false;

    if (!m_cmdpool->emplace<IGPUCommandPool::CDispatchIndirectCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IGPUBuffer>(buffer)))
        return false;

    return dispatchIndirect_impl(buffer, offset);
}

bool IGPUCommandBuffer::waitEvents(uint32_t eventCount, event_t* const* const pEvents, const SDependencyInfo* depInfo)
{
    if (eventCount == 0u)
        return false;

    for (uint32_t i = 0u; i < eventCount; ++i)
    {
        if (!pEvents[i] || !this->isCompatibleDevicewise(pEvents[i]))
            return false;
    }

    constexpr uint32_t MaxBarrierCount = 100u;
    assert(depInfo->memBarrierCount <= MaxBarrierCount);
    assert(depInfo->bufBarrierCount <= MaxBarrierCount);
    assert(depInfo->imgBarrierCount <= MaxBarrierCount);

    const IGPUBuffer* buffers_raw[MaxBarrierCount];
    for (auto i = 0; i < depInfo->bufBarrierCount; ++i)
        buffers_raw[i] = depInfo->bufBarriers[i].buffer.get();

    const IGPUImage* images_raw[MaxBarrierCount];
    for (auto i = 0; i < depInfo->imgBarrierCount; ++i)
        images_raw[i] = depInfo->imgBarriers[i].image.get();

    if (!m_cmdpool->emplace<IGPUCommandPool::CWaitEventsCmd>(m_segmentListHeadItr, m_segmentListTail, depInfo->bufBarrierCount, buffers_raw, depInfo->imgBarrierCount, images_raw, eventCount, pEvents))
        return false;

    return waitEvents_impl(eventCount, pEvents, depInfo);
}

bool IGPUCommandBuffer::drawMeshBuffer(const IGPUMeshBuffer::base_t* meshBuffer)
{
    if (meshBuffer && !meshBuffer->getInstanceCount())
        return false;

    const auto* pipeline = meshBuffer->getPipeline();
    const auto bindingFlags = pipeline->getVertexInputParams().enabledBindingFlags;
    auto vertexBufferBindings = meshBuffer->getVertexBufferBindings();
    auto indexBufferBinding = meshBuffer->getIndexBufferBinding();
    const auto indexType = meshBuffer->getIndexType();

    const nbl::video::IGPUBuffer* gpuBufferBindings[nbl::asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT];
    {
        for (size_t i = 0; i < nbl::asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; ++i)
            gpuBufferBindings[i] = vertexBufferBindings[i].buffer.get();
    }

    size_t bufferBindingsOffsets[nbl::asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT];
    {
        for (size_t i = 0; i < nbl::asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; ++i)
            bufferBindingsOffsets[i] = vertexBufferBindings[i].offset;
    }

    bindVertexBuffers(0, nbl::asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT, gpuBufferBindings, bufferBindingsOffsets);
    bindIndexBuffer(indexBufferBinding.buffer.get(), indexBufferBinding.offset, indexType);

    const bool isIndexed = indexType != nbl::asset::EIT_UNKNOWN;

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

bool IGPUCommandBuffer::copyBuffer(const buffer_t* srcBuffer, buffer_t* dstBuffer, uint32_t regionCount, const asset::SBufferCopy* pRegions)
{
    if (!srcBuffer || srcBuffer->getAPIType() != getAPIType())
        return false;

    if (!dstBuffer || dstBuffer->getAPIType() != getAPIType())
        return false;

    if (!this->isCompatibleDevicewise(srcBuffer))
        return false;

    if (!this->isCompatibleDevicewise(dstBuffer))
        return false;

    if (regionCount == 0u)
        return false;

    if (!m_cmdpool->emplace<IGPUCommandPool::CCopyBufferCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IGPUBuffer>(srcBuffer), core::smart_refctd_ptr<const IGPUBuffer>(dstBuffer)))
        return false;

    return copyBuffer_impl(srcBuffer, dstBuffer, regionCount, pRegions);
}

bool IGPUCommandBuffer::copyImage(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SImageCopy* pRegions)
{
    if (!srcImage || srcImage->getAPIType() != getAPIType())
        return false;

    if (!dstImage || dstImage->getAPIType() != getAPIType())
        return false;

    if (!this->isCompatibleDevicewise(srcImage))
        return false;

    if (!this->isCompatibleDevicewise(dstImage))
        return false;

    if (!dstImage->validateCopies(pRegions, pRegions + regionCount, srcImage))
        return false;

    if (!m_cmdpool->emplace<IGPUCommandPool::CCopyImageCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IGPUImage>(srcImage), core::smart_refctd_ptr<const IGPUImage>(dstImage)))
        return false;

    return copyImage_impl(srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);
}

bool IGPUCommandBuffer::copyBufferToImage(const buffer_t* srcBuffer, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions)
{
    if (!srcBuffer || srcBuffer->getAPIType() != getAPIType())
        return false;

    if (!dstImage || dstImage->getAPIType() != getAPIType())
        return false;

    if (!this->isCompatibleDevicewise(srcBuffer))
        return false;

    if (!this->isCompatibleDevicewise(dstImage))
        return false;

    if (!m_cmdpool->emplace<IGPUCommandPool::CCopyBufferToImageCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IGPUBuffer>(srcBuffer), core::smart_refctd_ptr<const IGPUImage>(dstImage)))
        return false;

    return copyBufferToImage_impl(srcBuffer, dstImage, dstImageLayout, regionCount, pRegions);
}

bool IGPUCommandBuffer::blitImage(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageBlit* pRegions, asset::ISampler::E_TEXTURE_FILTER filter)
{
    if (!srcImage || (srcImage->getAPIType() != getAPIType()))
        return false;

    if (!dstImage || (dstImage->getAPIType() != getAPIType()))
        return false;

    if (!this->isCompatibleDevicewise(srcImage))
        return false;

    if (!this->isCompatibleDevicewise(dstImage))
        return false;

    for (uint32_t i = 0u; i < regionCount; ++i)
    {
        if (pRegions[i].dstSubresource.aspectMask != pRegions[i].srcSubresource.aspectMask)
            return false;
        if (pRegions[i].dstSubresource.layerCount != pRegions[i].srcSubresource.layerCount)
            return false;
    }

    if (!m_cmdpool->emplace<IGPUCommandPool::CBlitImageCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IGPUImage>(srcImage), core::smart_refctd_ptr<const IGPUImage>(dstImage)))
        return false;

    return blitImage_impl(srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions, filter);
}

bool IGPUCommandBuffer::copyImageToBuffer(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, buffer_t* dstBuffer, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions)
{
    if (!srcImage || (srcImage->getAPIType() != getAPIType()))
        return false;

    if (!dstBuffer || (dstBuffer->getAPIType() != getAPIType()))
        return false;

    if (!this->isCompatibleDevicewise(srcImage))
        return false;

    if (!this->isCompatibleDevicewise(dstBuffer))
        return false;

    if (!m_cmdpool->emplace<IGPUCommandPool::CCopyImageToBufferCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IGPUImage>(srcImage), core::smart_refctd_ptr<const IGPUBuffer>(dstBuffer)))
        return false;

    return copyImageToBuffer_impl(srcImage, srcImageLayout, dstBuffer, regionCount, pRegions);
}

bool IGPUCommandBuffer::resolveImage(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageResolve* pRegions)
{
    if (!srcImage || srcImage->getAPIType() != getAPIType())
        return false;

    if (!dstImage || dstImage->getAPIType() != getAPIType())
        return false;

    if (!this->isCompatibleDevicewise(srcImage))
        return false;

    if (!this->isCompatibleDevicewise(dstImage))
        return false;

    if (!m_cmdpool->emplace<IGPUCommandPool::CResolveImageCmd>(m_segmentListHeadItr, m_segmentListTail, core::smart_refctd_ptr<const IGPUImage>(srcImage), core::smart_refctd_ptr<const IGPUImage>(dstImage)))
        return false;

    return resolveImage_impl(srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);
}

bool IGPUCommandBuffer::executeCommands(uint32_t count, cmdbuf_t* const* const cmdbufs)
{
    for (uint32_t i = 0u; i < count; ++i)
    {
        if (!cmdbufs[i] || (cmdbufs[i]->getLevel() != EL_SECONDARY))
            return false;

        if (!this->isCompatibleDevicewise(cmdbufs[i]))
            return false;
    }

    auto commandBuffers = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<core::smart_refctd_ptr<const IGPUCommandBuffer>>>(count);
    for (auto i = 0; i < commandBuffers->size(); ++i)
        commandBuffers->begin()[i] = core::smart_refctd_ptr<const IGPUCommandBuffer>(cmdbufs[i]);

    if (!m_cmdpool->emplace<IGPUCommandPool::CExecuteCommandsCmd>(m_segmentListHeadItr, m_segmentListTail, std::move(commandBuffers)))
        return false;

    return executeCommands_impl(count, cmdbufs);
}

bool IGPUCommandBuffer::regenerateMipmaps(IGPUImage* img, uint32_t lastReadyMip, asset::IImage::E_ASPECT_FLAGS aspect)
{
    const auto apiType = getAPIType();
    if ((apiType == EAT_OPENGL) || (apiType == EAT_OPENGL_ES))
        return regenerateMipmaps_impl(img, lastReadyMip, aspect);

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

}