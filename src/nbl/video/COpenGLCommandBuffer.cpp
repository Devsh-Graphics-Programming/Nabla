#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/IOpenGL_LogicalDevice.h"

#include "nbl/video/COpenGLCommandBuffer.h"
#include "nbl/video/COpenGLQueryPool.h"
#include "nbl/video/COpenGLCommon.h"

namespace nbl::video
{

COpenGLCommandBuffer::COpenGLCommandBuffer(core::smart_refctd_ptr<const ILogicalDevice>&& dev, E_LEVEL lvl, core::smart_refctd_ptr<IGPUCommandPool>&& _cmdpool, system::logger_opt_smart_ptr&& logger, const COpenGLFeatureMap* _features)
    : IGPUCommandBuffer(std::move(dev), lvl, std::move(_cmdpool), std::move(logger)), m_features(_features)
{
    // default values tracked by engine
    m_stateCache.nextState.rasterParams.multisampleEnable = 0;
    m_stateCache.nextState.rasterParams.depthFunc = GL_GEQUAL;
    m_stateCache.nextState.rasterParams.frontFace = GL_CCW;
}

bool COpenGLCommandBuffer::begin_impl(core::bitflag<E_USAGE> flags, const SInheritanceInfo* inheritanceInfo)
{
    m_GLSegmentListHeadItr.m_segment = nullptr;
    m_GLSegmentListHeadItr.m_cmd = nullptr;
    m_GLSegmentListTail = nullptr;

    return true;
}

void COpenGLCommandBuffer::releaseResourcesBackToPool_impl()
{
    m_cmdpool->deleteCommandSegmentList(m_GLSegmentListHeadItr, m_GLSegmentListTail);
}

bool COpenGLCommandBuffer::beginRenderpass_clearAttachments(SOpenGLContextIndependentCache* stateCache, const SRenderpassBeginInfo& info, const system::logger_opt_ptr logger, IGPUCommandPool* cmdpool, IGPUCommandPool::CCommandSegment::Iterator& segmentListHeadItr, IGPUCommandPool::CCommandSegment*& segmentListTail, const E_API_TYPE apiType, const COpenGLFeatureMap* features)
{
    auto& rp = info.framebuffer->getCreationParameters().renderpass;
    auto& sub = rp->getSubpasses().begin()[0];
    auto* color = sub.colorAttachments;
    auto* depthstencil = sub.depthStencilAttachment;
    auto* descriptions = rp->getAttachments().begin();

    for (uint32_t i = 0u; i < sub.colorAttachmentCount; ++i)
    {
        const uint32_t a = color[i].attachment;

        if (descriptions[a].loadOp == asset::IRenderpass::ELO_CLEAR)
        {
            if (a < info.clearValueCount)
            {
                asset::E_FORMAT fmt = descriptions[a].format;
                if (!cmdpool->emplace<COpenGLCommandPool::CClearNamedFramebufferCmd>(segmentListHeadItr, segmentListTail, stateCache->currentState.framebuffer.hash, fmt, GL_COLOR, info.clearValues[a], i))
                    return false;
            }
            else
            {
                logger.log("Begin renderpass command: not enough clear values provided, an attachment not cleared!", system::ILogger::ELL_ERROR);
            }
        }
    }

    if (depthstencil)
    {
        auto* depthstencilDescription = descriptions + depthstencil->attachment;
        if (depthstencilDescription->loadOp == asset::IRenderpass::ELO_CLEAR)
        {
            if (depthstencil->attachment < info.clearValueCount)
            {
                asset::E_FORMAT fmt = depthstencilDescription->format;

                // isnt there a way in vulkan to clear only depth or only stencil part?? TODO

                const bool is_depth = asset::isDepthOnlyFormat(fmt);
                const bool is_stencil = asset::isStencilOnlyFormat(fmt);
                const bool is_depth_stencil = asset::isDepthOrStencilFormat(fmt);

                GLenum bufferType = 0;
                {
                    if (is_depth)
                        bufferType = GL_DEPTH;
                    else if (is_stencil)
                        bufferType = GL_STENCIL;
                    else if (is_depth_stencil)
                        bufferType = GL_DEPTH_STENCIL;
                }

                const auto stateBackup = stateCache->backupAndFlushStateClear(cmdpool, segmentListHeadItr, segmentListTail, false, (is_depth || is_depth_stencil), (is_stencil || is_depth_stencil), apiType, features);
                const bool cmdEmplaceFailed = !cmdpool->emplace<COpenGLCommandPool::CClearNamedFramebufferCmd>(segmentListHeadItr, segmentListTail, stateCache->currentState.framebuffer.hash, fmt, bufferType, info.clearValues[depthstencil->attachment], 0);
                stateCache->restoreStateAfterClear(stateBackup);

                if (cmdEmplaceFailed)
                    return false;
            }
            else
            {
                logger.log("Begin renderpass command: not enough clear values provided, an attachment not cleared!", system::ILogger::ELL_ERROR);
            }
        }
    }

    return true;
}

bool COpenGLCommandBuffer::pushConstants_validate(const IGPUPipelineLayout* _layout, uint32_t _stages, uint32_t _offset, uint32_t _size, const void* _values)
{
    if (!_layout || !_values)
        return false;
    if (!_size)
        return false;
    if (!_stages)
        return false;
    if (!core::is_aligned_to(_offset, 4u))
        return false;
    if (!core::is_aligned_to(_size, 4u))
        return false;
    if (_offset >= IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE)
        return false;
    if ((_offset + _size) > IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE)
        return false;

    asset::SPushConstantRange updateRange;
    updateRange.offset = _offset;
    updateRange.size = _size;

#ifdef _NBL_DEBUG
    //TODO validation:
    /*
    For each byte in the range specified by offset and size and for each shader stage in stageFlags,
    there must be a push constant range in layout that includes that byte and that stage
    */
    for (const auto& rng : _layout->getPushConstantRanges())
    {
        /*
        For each byte in the range specified by offset and size and for each push constant range that overlaps that byte,
        stageFlags must include all stages in that push constant ranges VkPushConstantRange::stageFlags
        */
        if (updateRange.overlap(rng) && ((_stages & rng.stageFlags) != rng.stageFlags))
            return false;
    }
#endif//_NBL_DEBUG

    return true;
}

void COpenGLCommandBuffer::executeAll(IOpenGL_FunctionTable* gl, SOpenGLContextDependentCache& queueLocal, uint32_t ctxid) const
{
    IGPUCommandPool::CCommandSegment::Iterator itr = m_GLSegmentListHeadItr;

    if (itr.m_segment && itr.m_cmd)
    {
        while (itr.m_cmd->getSize() != 0u)
        {
            auto* glcmd = static_cast<COpenGLCommandPool::IOpenGLCommand*>(itr.m_cmd);
            glcmd->operator()(gl, queueLocal, ctxid, m_logger.getOptRawPtr());

            itr.m_cmd = reinterpret_cast<IGPUCommandPool::ICommand*>(reinterpret_cast<uint8_t*>(itr.m_cmd) + itr.m_cmd->getSize());

            // We potentially continue to the next command segment under any one of the two conditions:
            const bool potentiallyContinueToNextSegment =
                // 1. If the we run past the storage of the current segment.
                ((reinterpret_cast<uint8_t*>(itr.m_cmd) - itr.m_segment->getData()) >= IGPUCommandPool::CCommandSegment::STORAGE_SIZE)
                ||
                // 2. If we encounter a 0-sized command (terminating command) before running out of the current segment. This case will arise when the current
                // segment doesn't have enough storage to hold the next command.
                (itr.m_cmd->getSize() == 0);
            if (potentiallyContinueToNextSegment)
            {
                IGPUCommandPool::CCommandSegment* nextSegment = itr.m_segment->getNext();
                if (!nextSegment)
                    break;

                itr.m_segment = nextSegment;
                itr.m_cmd = itr.m_segment->getFirstCommand();
            }
        }
    }
}

bool COpenGLCommandBuffer::bindGraphicsPipeline_impl(const graphics_pipeline_t* pipeline)
{
    m_stateCache.updateNextState_pipelineAndRaster(pipeline, m_cmdpool.get(), m_GLSegmentListHeadItr, m_GLSegmentListTail);

    auto* rpindependent = pipeline->getRenderpassIndependentPipeline();
    auto* glppln = static_cast<const COpenGLRenderpassIndependentPipeline*>(rpindependent);
    m_stateCache.nextState.vertexInputParams.vaokey = glppln->getVAOHash();

    return true;
}

void COpenGLCommandBuffer::bindComputePipeline_impl(const compute_pipeline_t* pipeline)
{
    m_stateCache.nextState.pipeline.compute.pipeline = static_cast<const COpenGLComputePipeline*>(pipeline);
}

bool COpenGLCommandBuffer::resetQueryPool_impl(IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount)
{
    auto* gl_queryPool = static_cast<COpenGLQueryPool*>(queryPool);
    gl_queryPool->resetQueries(firstQuery, queryCount, m_cmdpool.get(), m_GLSegmentListHeadItr, m_GLSegmentListTail);

    return true;
}

bool COpenGLCommandBuffer::beginQuery_impl(IQueryPool* queryPool, uint32_t query, core::bitflag<video::IQueryPool::E_QUERY_CONTROL_FLAGS> flags)
{
    const COpenGLQueryPool* qp = static_cast<const COpenGLQueryPool*>(queryPool);
    auto currentQuery = core::bitflag(qp->getCreationParameters().queryType);
    if (!queriesActive.hasFlags(currentQuery))
    {
        if (!qp->beginQuery(query, flags.value, m_cmdpool.get(), m_GLSegmentListHeadItr, m_GLSegmentListTail))
            return false;

        queriesActive |= currentQuery;

        uint32_t queryTypeIndex = std::log2<uint32_t>(currentQuery.value);
        currentlyRecordingQueries[queryTypeIndex] = std::make_tuple(qp, query, currentlyRecordingRenderPass, 0u);
    }
    else
    {
        assert(false); // There is an active query with the same query type.
        return false;
    }

    return true;
}

bool COpenGLCommandBuffer::writeTimestamp_impl(asset::E_PIPELINE_STAGE_FLAGS pipelineStage, IQueryPool* queryPool, uint32_t query)
{
    auto* gl_queryPool = static_cast<COpenGLQueryPool*>(queryPool);
    if (!gl_queryPool->writeTimestamp(query, m_cmdpool.get(), m_GLSegmentListHeadItr, m_GLSegmentListTail))
        return false;

    return true;
}

bool COpenGLCommandBuffer::endQuery_impl(IQueryPool* queryPool, uint32_t query)
{
    COpenGLQueryPool* qp = static_cast<COpenGLQueryPool*>(queryPool);
    auto currentQuery = core::bitflag(qp->getCreationParameters().queryType);
    if (queriesActive.hasFlags(currentQuery))
    {
        uint32_t queryTypeIndex = std::log2<uint32_t>(currentQuery.value);
        IQueryPool const* currentQueryPool = std::get<0>(currentlyRecordingQueries[queryTypeIndex]);
        uint32_t currentQueryIndex = std::get<1>(currentlyRecordingQueries[queryTypeIndex]);
        renderpass_t const* currentQueryRenderpass = std::get<2>(currentlyRecordingQueries[queryTypeIndex]);
        uint32_t currentQuerySubpassIndex = std::get<3>(currentlyRecordingQueries[queryTypeIndex]);

        if (currentQueryPool != queryPool || currentQueryIndex != query)
        {
            assert(false); // You must end the same query you began for every query type.
            return false;
        }
        if (currentQueryRenderpass != currentlyRecordingRenderPass)
        {
            assert(false); // Query either starts and ends in the same subpass or starts and ends entirely outside a renderpass
            return false;
        }

        // currentlyRecordingQuery assert tuple -> same query index and query pool -> same renderpass
        if (!qp->endQuery(query, m_cmdpool.get(), m_GLSegmentListHeadItr, m_GLSegmentListTail))
            return false;

        queriesActive &= ~currentQuery;
    }
    else
    {
        assert(false); // QueryType was not active to end.
        return false;
    }

    return true;
}

bool COpenGLCommandBuffer::copyQueryPoolResults_impl(IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, buffer_t* dstBuffer, size_t dstOffset, size_t stride, core::bitflag<video::IQueryPool::E_QUERY_RESULTS_FLAGS> flags)
{
    const IPhysicalDevice* physdev = getOriginDevice()->getPhysicalDevice();
    // if (!physdev->getFeatures().allowCommandBufferQueryCopies)
    // {
    //     assert(false); // allowCommandBufferQueryCopies feature not enabled -> can't write query results to buffer
    //     return false;
    // }

    const COpenGLBuffer* buffer = static_cast<const COpenGLBuffer*>(dstBuffer);
    const COpenGLQueryPool* qp = IBackendObject::compatibility_cast<const COpenGLQueryPool*>(queryPool, this);
    auto queryPoolQueriesCount = qp->getCreationParameters().queryCount;

    const GLuint bufferId = buffer->getOpenGLName();

    IQueryPool::E_QUERY_TYPE queryType = qp->getCreationParameters().queryType;
    assert(queryType == IQueryPool::E_QUERY_TYPE::EQT_OCCLUSION || queryType == IQueryPool::E_QUERY_TYPE::EQT_TIMESTAMP);

    const bool use64Version = flags.hasFlags(IQueryPool::E_QUERY_RESULTS_FLAGS::EQRF_64_BIT);
    const bool availabilityFlag = flags.hasFlags(IQueryPool::E_QUERY_RESULTS_FLAGS::EQRF_WITH_AVAILABILITY_BIT);
    const bool waitForAllResults = flags.hasFlags(IQueryPool::E_QUERY_RESULTS_FLAGS::EQRF_WAIT_BIT);
    const bool partialResults = flags.hasFlags(IQueryPool::E_QUERY_RESULTS_FLAGS::EQRF_PARTIAL_BIT);

    if (firstQuery + queryCount > queryPoolQueriesCount)
    {
        m_logger.log("The sum of firstQuery and queryCount must be less than or equal to the number of queries in queryPool", system::ILogger::ELL_ERROR);
        return false;
    }
    if (partialResults && queryType == IQueryPool::E_QUERY_TYPE::EQT_TIMESTAMP)
    {
        m_logger.log("QUERY_RESULT_PARTIAL_BIT must not be used if the pool�s queryType is QUERY_TYPE_TIMESTAMP.", system::ILogger::ELL_ERROR);
        return false;
    }

    size_t currentDataPtrOffset = dstOffset;
    const uint32_t glQueriesPerQuery = qp->getGLQueriesPerQuery();
    const size_t queryElementDataSize = (use64Version) ? sizeof(GLuint64) : sizeof(GLuint); // each query might write to multiple values/elements
    const size_t eachQueryDataSize = queryElementDataSize * glQueriesPerQuery;
    const size_t eachQueryWithAvailabilityDataSize = (availabilityFlag) ? queryElementDataSize + eachQueryDataSize : eachQueryDataSize;

    const size_t bufferDataSize = buffer->getSize();

    assert(core::is_aligned_to(dstOffset, queryElementDataSize));
    assert(stride >= eachQueryWithAvailabilityDataSize);
    assert(stride && core::is_aligned_to(stride, eachQueryWithAvailabilityDataSize)); // stride must be aligned to each query data size considering the specified flags
    assert((bufferDataSize - currentDataPtrOffset) >= (queryCount * stride)); // bufferDataSize is not enough for "queryCount" queries and specified stride
    assert((bufferDataSize - currentDataPtrOffset) >= (queryCount * eachQueryWithAvailabilityDataSize)); // bufferDataSize is not enough for "queryCount" queries with considering the specified flags

    // iterate on each query
    for (uint32_t i = 0; i < queryCount; ++i)
    {
        if (currentDataPtrOffset >= bufferDataSize)
        {
            assert(false);
            break;
        }

        const size_t queryDataOffset = currentDataPtrOffset;
        const size_t availabilityDataOffset = queryDataOffset + eachQueryDataSize; // Write Availability to this offset if flag specified

        // iterate on each gl query (we may have multiple gl queries per query like pipelinestatistics query type)
        const uint32_t queryIndex = i + firstQuery;
        const uint32_t glQueryBegin = queryIndex * glQueriesPerQuery;
        bool allGlQueriesAvailable = true;

        for (uint32_t q = 0; q < glQueriesPerQuery; ++q)
        {
            const size_t subQueryDataOffset = queryDataOffset + q * queryElementDataSize;
            const uint32_t queryIdx = glQueryBegin + q;

            GLenum pname;
            if (waitForAllResults)
            {
                // Has WAIT_BIT -> Get Result with Wait (GL_QUERY_RESULT) + don't getQueryAvailability (if availability flag is set it will report true)
                pname = GL_QUERY_RESULT;
            }
            else if (partialResults)
            {
                // Has PARTIAL_BIT but no WAIT_BIT -> (read vk spec) -> result value between zero and the final result value
                // No PARTIAL queries for GL -> GL_QUERY_RESULT_NO_WAIT best match
                pname = GL_QUERY_RESULT_NO_WAIT;
            }
            else if (availabilityFlag)
            {
                // Only Availablity -> Get Results with NoWait + get Query Availability
                pname = GL_QUERY_RESULT_NO_WAIT;
            }
            else
            {
                // No Flags -> GL_QUERY_RESULT_NO_WAIT
                pname = GL_QUERY_RESULT_NO_WAIT;
            }

            if (availabilityFlag && !waitForAllResults && (q == glQueriesPerQuery - 1))
            {
                if (!m_cmdpool->emplace<COpenGLCommandPool::CGetQueryBufferObjectUICmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, qp, queryIdx, use64Version, bufferId, GL_QUERY_RESULT_AVAILABLE, availabilityDataOffset))
                    return false;
            }

            if (!m_cmdpool->emplace<COpenGLCommandPool::CGetQueryBufferObjectUICmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, qp, queryIdx, use64Version, bufferId, pname, subQueryDataOffset))
                return false;

            if (availabilityFlag && waitForAllResults && (q == glQueriesPerQuery - 1))
            {
                if (!m_cmdpool->emplace<COpenGLCommandPool::CGetQueryBufferObjectUICmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, qp, queryIdx, use64Version, bufferId, GL_QUERY_RESULT_AVAILABLE, availabilityDataOffset))
                    return false;
            }
        }

        currentDataPtrOffset += stride;
    }

    return true;
}

bool COpenGLCommandBuffer::bindDescriptorSets_impl(asset::E_PIPELINE_BIND_POINT pipelineBindPoint, const pipeline_layout_t* layout_, uint32_t firstSet_, const uint32_t descriptorSetCount_,
    const descriptor_set_t* const* const descriptorSets_, const uint32_t dynamicOffsetCount_, const uint32_t* dynamicOffsets_)
{
    uint32_t firstSet = firstSet_;
    uint32_t dsCount = 0u;
    core::smart_refctd_ptr<const IGPUDescriptorSet> descriptorSets[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT];

    constexpr uint32_t MaxDynamicOffsets = SOpenGLState::MaxDynamicOffsets * IGPUPipelineLayout::DESCRIPTOR_SET_COUNT;
    uint32_t dynamicOffsets[MaxDynamicOffsets];
    uint32_t dynamicOffsetCount = 0u;

    auto dynamicOffsetsIt = dynamicOffsets_;

    asset::E_PIPELINE_BIND_POINT pbp = pipelineBindPoint;

    // Will bind non-null [firstSet, dsCount) ranges with one call
    auto bind = [this, &descriptorSets, pbp, layout_, dynamicOffsets, dynamicOffsetCount](const uint32_t first, const uint32_t count)
    {
        const IGPUPipelineLayout* layouts[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT]{};
        for (uint32_t j = 0u; j < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++j)
            layouts[j] = m_stateCache.nextState.descriptorsParams[pbp].descSets[j].pplnLayout.get();

        const IGPUDescriptorSet* descriptorSets_raw[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT]{};
        for (uint32_t j = 0u; j < count; ++j)
            descriptorSets_raw[j] = descriptorSets[j].get();

        bindDescriptorSets_generic(layout_, first, count, descriptorSets_raw, layouts);

        for (uint32_t j = 0u; j < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++j)
        {
            if (!layouts[j])
                m_stateCache.nextState.descriptorsParams[pbp].descSets[j] = { nullptr, nullptr, {}, 0u }; // TODO: have a default constructor that makes sense and prevents us from screwing up
        }

        uint32_t offsetOfDynamicOffsets = 0u;
        for (uint32_t i = 0u; i < count; i++)
        {
            auto glDS = static_cast<const COpenGLDescriptorSet*>(descriptorSets_raw[i]);
            if (glDS)
            {
                const auto dynamicOffsetCount = glDS->getDynamicOffsetCount();

                auto& stateDS = m_stateCache.nextState.descriptorsParams[pbp].descSets[first + i];
                stateDS.pplnLayout = core::smart_refctd_ptr<const COpenGLPipelineLayout>(static_cast<const COpenGLPipelineLayout*>(layout_));
                stateDS.set = core::smart_refctd_ptr<const COpenGLDescriptorSet>(glDS);
                std::copy_n(dynamicOffsets + offsetOfDynamicOffsets, dynamicOffsetCount, stateDS.dynamicOffsets);
                stateDS.revision = glDS->getRevision();

                offsetOfDynamicOffsets += dynamicOffsetCount;
            }
        }
        assert(offsetOfDynamicOffsets == dynamicOffsetCount);
    };

    for (auto i = 0; i < descriptorSetCount_; ++i)
    {
        if (descriptorSets_[i])
        {
            descriptorSets[dsCount++] = core::smart_refctd_ptr<const IGPUDescriptorSet>(descriptorSets_[i]);
            const auto count = IBackendObject::compatibility_cast<const COpenGLDescriptorSet*>(descriptorSets_[i], this)->getDynamicOffsetCount();
            std::copy_n(dynamicOffsetsIt, count, dynamicOffsets + dynamicOffsetCount);
            dynamicOffsetCount += count;
            dynamicOffsetsIt += count;
            continue;
        }

        if (dsCount)
            bind(firstSet, dsCount);

        {
            firstSet = firstSet_ + 1u + i;
            dsCount = 0u;
            dynamicOffsetCount = 0u;
        }
    }

    // TODO(achal): This shouldn't come after we change the m_stateCache, should it?
    if ((dynamicOffsetsIt - dynamicOffsets_) != dynamicOffsetCount_)
    {
        m_logger.log("IGPUCommandBuffer::bindDescriptorSets failed, `dynamicOffsetCount` does not match the number of dynamic offsets required by the descriptor set layouts!", system::ILogger::ELL_ERROR);
        return false;
    }

    bind(firstSet, dsCount);

    return true;
}

bool COpenGLCommandBuffer::pushConstants_impl(const pipeline_layout_t* layout, core::bitflag<asset::IShader::E_SHADER_STAGE> stageFlags, uint32_t offset, uint32_t size, const void* pValues)
{
    if (pushConstants_validate(layout, stageFlags.value, offset, size, pValues))
    {
        const auto* gl_pipelineLayout = static_cast<const COpenGLPipelineLayout*>(layout);

        if (stageFlags.value & asset::IShader::ESS_ALL_GRAPHICS)
        {
            if (!m_cmdpool->emplace<COpenGLCommandPool::CPushConstantsCmd<asset::EPBP_GRAPHICS>>(m_GLSegmentListHeadItr, m_GLSegmentListTail, gl_pipelineLayout, stageFlags, offset, size, pValues))
                return false;
        }

        if (stageFlags.value & asset::IShader::ESS_COMPUTE)
        {
            if (!m_cmdpool->emplace<COpenGLCommandPool::CPushConstantsCmd<asset::EPBP_COMPUTE>>(m_GLSegmentListHeadItr, m_GLSegmentListTail, gl_pipelineLayout, stageFlags, offset, size, pValues))
                return false;
        }
    }

    return true;
}

bool COpenGLCommandBuffer::clearColorImage_impl(image_t* image, asset::IImage::E_LAYOUT imageLayout, const asset::SClearColorValue* pColor, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges)
{
    const auto state_backup = m_stateCache.backupAndFlushStateClear(m_cmdpool.get(), m_GLSegmentListHeadItr, m_GLSegmentListTail, true, false, false, getAPIType(), m_features);

    bool anyFailed = false;
    for (uint32_t i = 0u; i < rangeCount; ++i)
    {
        auto& info = pRanges[i];

        for (uint32_t m = 0u; m < info.levelCount; ++m)
        {
            for (uint32_t l = 0u; l < info.layerCount; ++l)
            {
                if (!m_cmdpool->emplace<COpenGLCommandPool::CClearColorImageCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, static_cast<const COpenGLImage*>(image), info.baseMipLevel + m, info.baseArrayLayer + l, *pColor))
                    anyFailed = true;
            }
        }
    }

    m_stateCache.restoreStateAfterClear(state_backup);

    return !anyFailed;
}

bool COpenGLCommandBuffer::clearDepthStencilImage_impl(image_t* image, asset::IImage::E_LAYOUT imageLayout, const asset::SClearDepthStencilValue* pDepthStencil, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges)
{
    const auto fmt = image->getCreationParameters().format;

    const bool is_depth = asset::isDepthOnlyFormat(fmt);
    bool is_stencil = false;
    bool is_depth_stencil = false;
    if (!is_depth)
    {
        is_stencil = asset::isStencilOnlyFormat(fmt);
        if (!is_stencil)
            is_depth_stencil = asset::isDepthOrStencilFormat(fmt);
    }

    const auto state_backup = m_stateCache.backupAndFlushStateClear(m_cmdpool.get(), m_GLSegmentListHeadItr, m_GLSegmentListTail, false, (is_depth || is_depth_stencil), (is_stencil || is_depth_stencil), getAPIType(), m_features);

    bool anyFailed = false;
    for (uint32_t i = 0u; i < rangeCount; ++i)
    {
        const auto& info = pRanges[i];

        for (uint32_t m = 0u; m < info.levelCount; ++m)
        {
            for (uint32_t l = 0u; l < info.layerCount; ++l)
            {
                if (!m_cmdpool->emplace<COpenGLCommandPool::CClearDepthStencilImageCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, static_cast<const COpenGLImage*>(image), info.baseMipLevel + m, info.baseArrayLayer + l, *pDepthStencil))
                    anyFailed = true;
            }
        }
    }

    m_stateCache.restoreStateAfterClear(state_backup);

    return !anyFailed;
}

bool COpenGLCommandBuffer::clearAttachments(uint32_t attachmentCount, const asset::SClearAttachment* pAttachments, uint32_t rectCount, const asset::SClearRect* pRects)
{
    if (attachmentCount == 0u || rectCount == 0u)
        return false;

    auto& framebuffer = m_stateCache.currentState.framebuffer.fbo;
    if (!framebuffer)
        return false;

    auto& rp = framebuffer->getCreationParameters().renderpass;
    auto& sub = rp->getSubpasses().begin()[0];
    auto* color = sub.colorAttachments;
    auto* depthstencil = sub.depthStencilAttachment;
    auto* descriptions = rp->getAttachments().begin();

    bool anyFailed = false;
    for (uint32_t i = 0u; i < attachmentCount; ++i)
    {
        auto& attachment = pAttachments[i];
        if (attachment.aspectMask & asset::IImage::EAF_COLOR_BIT)
        {
            uint32_t num = attachment.colorAttachment;
            uint32_t a = color[num].attachment;
            asset::E_FORMAT fmt = descriptions[a].format;

            if (!m_cmdpool->emplace<COpenGLCommandPool::CClearNamedFramebufferCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, m_stateCache.currentState.framebuffer.hash, fmt, GL_COLOR, attachment.clearValue, num))
                anyFailed = true;
        }
        else if (attachment.aspectMask & (asset::IImage::EAF_DEPTH_BIT | asset::IImage::EAF_STENCIL_BIT))
        {
            GLenum bufferType = 0;
            {
                auto aspectMask = (attachment.aspectMask & (asset::IImage::EAF_DEPTH_BIT | asset::IImage::EAF_STENCIL_BIT));
                if (aspectMask == (asset::IImage::EAF_DEPTH_BIT | asset::IImage::EAF_STENCIL_BIT))
                    bufferType = GL_DEPTH_STENCIL;
                else if (aspectMask == asset::IImage::EAF_DEPTH_BIT)
                    bufferType = GL_DEPTH;
                else if (aspectMask == asset::IImage::EAF_STENCIL_BIT)
                    bufferType = GL_STENCIL;
            }

            if (!m_cmdpool->emplace<COpenGLCommandPool::CClearNamedFramebufferCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, m_stateCache.currentState.framebuffer.hash, asset::EF_UNKNOWN, bufferType, attachment.clearValue, 0))
                anyFailed = true;
        }
    }

    return !anyFailed;
}

bool COpenGLCommandBuffer::fillBuffer_impl(buffer_t* dstBuffer, size_t dstOffset, size_t size, uint32_t data)
{
    GLuint buf = static_cast<const COpenGLBuffer*>(dstBuffer)->getOpenGLName();

    if (!m_cmdpool->emplace<COpenGLCommandPool::CClearNamedBufferSubDataCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, buf, GL_R32UI, dstOffset, size, GL_RED, GL_UNSIGNED_INT, data))
        return false;

    return true;
}

bool COpenGLCommandBuffer::updateBuffer_impl(buffer_t* dstBuffer, size_t dstOffset, size_t dataSize, const void* pData)
{
    GLuint buf = static_cast<const COpenGLBuffer*>(dstBuffer)->getOpenGLName();

    if (!m_cmdpool->emplace<COpenGLCommandPool::CNamedBufferSubDataCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, buf, dstOffset, dataSize, pData))
        return false;

    return true;
}

void COpenGLCommandBuffer::bindVertexBuffers_impl(uint32_t firstBinding, uint32_t bindingCount, const buffer_t* const* const pBuffers, const size_t* pOffsets)
{
    for (uint32_t i = 0u; i < bindingCount; ++i)
    {
        auto& binding = m_stateCache.nextState.vertexInputParams.vaoval.vtxBindings[firstBinding + i];
        binding.buffer = core::smart_refctd_ptr<const COpenGLBuffer>(static_cast<const COpenGLBuffer*>(pBuffers[i]));
        binding.offset = pOffsets[i];
    }
}

bool COpenGLCommandBuffer::setScissor(uint32_t firstScissor, uint32_t scissorCount, const VkRect2D* pScissors)
{
    // TODO ?
    return true;
}

bool COpenGLCommandBuffer::setDepthBounds(float minDepthBounds, float maxDepthBounds)
{
    // TODO ?
    return true;
}

bool COpenGLCommandBuffer::setStencilCompareMask(asset::E_STENCIL_FACE_FLAGS faceMask, uint32_t compareMask)
{
    if (faceMask & asset::ESFF_FRONT_BIT)
        m_stateCache.nextState.rasterParams.stencilFunc_front.mask = compareMask;
    if (faceMask & asset::ESFF_BACK_BIT)
        m_stateCache.nextState.rasterParams.stencilFunc_back.mask = compareMask;

    return true;
}

bool COpenGLCommandBuffer::setStencilWriteMask(asset::E_STENCIL_FACE_FLAGS faceMask, uint32_t writeMask)
{
    if (faceMask & asset::ESFF_FRONT_BIT)
        m_stateCache.nextState.rasterParams.stencilWriteMask_front = writeMask;
    if (faceMask & asset::ESFF_BACK_BIT)
        m_stateCache.nextState.rasterParams.stencilWriteMask_back = writeMask;

    return true;
}

bool COpenGLCommandBuffer::setStencilReference(asset::E_STENCIL_FACE_FLAGS faceMask, uint32_t reference)
{
    if (faceMask & asset::ESFF_FRONT_BIT)
        m_stateCache.nextState.rasterParams.stencilFunc_front.ref = reference;
    if (faceMask & asset::ESFF_BACK_BIT)
        m_stateCache.nextState.rasterParams.stencilFunc_back.ref = reference;

    return true;
}

bool COpenGLCommandBuffer::dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
    if (!m_stateCache.flushStateCompute(SOpenGLContextIndependentCache::GSB_ALL, m_cmdpool.get(), m_GLSegmentListHeadItr, m_GLSegmentListTail, m_features))
        return false;

    if (!m_cmdpool->emplace<COpenGLCommandPool::CDispatchComputeCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, groupCountX, groupCountY, groupCountZ))
        return false;

    return true;
}

bool COpenGLCommandBuffer::dispatchIndirect_impl(const buffer_t* buffer, size_t offset)
{
    m_stateCache.nextState.dispatchIndirect.buffer = core::smart_refctd_ptr<const COpenGLBuffer>(static_cast<const COpenGLBuffer*>(buffer));

    if (!m_stateCache.flushStateCompute(SOpenGLContextIndependentCache::GSB_ALL, m_cmdpool.get(), m_GLSegmentListHeadItr, m_GLSegmentListTail, m_features))
        return false;

    if (!m_cmdpool->emplace<COpenGLCommandPool::CDispatchComputeIndirectCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, static_cast<GLintptr>(offset)))
        return false;

    return true;
}

bool COpenGLCommandBuffer::dispatchBase(uint32_t baseGroupX, uint32_t baseGroupY, uint32_t baseGroupZ, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
    // no such thing in opengl (easy to emulate tho)
    // maybe spirv-cross emits some uniforms for this?
    return true;
}

bool COpenGLCommandBuffer::setEvent_impl(event_t* _event, const SDependencyInfo& depInfo)
{
    //https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdSetEvent2KHR.html
    // A memory dependency is defined between the event signal operation and commands that occur earlier in submission order.
    return true;
}

bool COpenGLCommandBuffer::resetEvent_impl(event_t* _event, asset::E_PIPELINE_STAGE_FLAGS stageMask)
{
    // currently no-op
    return true;
}

bool COpenGLCommandBuffer::waitEvents_impl(uint32_t eventCount, event_t* const* const pEvents, const SDependencyInfo* depInfo)
{
    GLbitfield barrier = 0;
    for (uint32_t i = 0u; i < eventCount; ++i)
    {
        auto& dep = depInfo[i];
        barrier |= barriersToMemBarrierBits(SOpenGLBarrierHelper(m_features), dep.memBarrierCount, dep.memBarriers, dep.bufBarrierCount, dep.bufBarriers, dep.imgBarrierCount, dep.imgBarriers);
    }

    if (!m_cmdpool->emplace<COpenGLCommandPool::CMemoryBarrierCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, barrier))
        return false;

    return true;
}

bool COpenGLCommandBuffer::copyBuffer_impl(const buffer_t* srcBuffer, buffer_t* dstBuffer, uint32_t regionCount, const asset::SBufferCopy* pRegions)
{
    GLuint readb = static_cast<const COpenGLBuffer*>(srcBuffer)->getOpenGLName();
    GLuint writeb = static_cast<const COpenGLBuffer*>(dstBuffer)->getOpenGLName();
    for (uint32_t i = 0u; i < regionCount; ++i)
    {
        const asset::SBufferCopy& cp = pRegions[i];
        if (!m_cmdpool->emplace<COpenGLCommandPool::CCopyNamedBufferSubDataCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, readb, writeb, cp.srcOffset, cp.dstOffset, cp.size))
            return false;
    }

    return true;
}

bool COpenGLCommandBuffer::copyImage_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SImageCopy* pRegions)
{
    auto src = static_cast<const COpenGLImage*>(srcImage);
    auto dst = static_cast<COpenGLImage*>(dstImage);

    IGPUImage::E_TYPE srcType = srcImage->getCreationParameters().type;
    IGPUImage::E_TYPE dstType = dstImage->getCreationParameters().type;

    constexpr GLenum type2Target[3u] = { GL_TEXTURE_1D_ARRAY,GL_TEXTURE_2D_ARRAY,GL_TEXTURE_3D };

    for (auto it = pRegions; it != pRegions + regionCount; it++)
    {
        if (!m_cmdpool->emplace<COpenGLCommandPool::CCopyImageSubDataCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail,
            src->getOpenGLName(),
            type2Target[srcType],
            it->srcSubresource.mipLevel,
            it->srcOffset.x,
            srcType == IGPUImage::ET_1D ? it->srcSubresource.baseArrayLayer : it->srcOffset.y,
            srcType == IGPUImage::ET_2D ? it->srcSubresource.baseArrayLayer : it->srcOffset.z,
            dst->getOpenGLName(),
            type2Target[dstType],
            it->dstSubresource.mipLevel,
            it->dstOffset.x,
            dstType == IGPUImage::ET_1D ? it->dstSubresource.baseArrayLayer : it->dstOffset.y,
            dstType == IGPUImage::ET_2D ? it->dstSubresource.baseArrayLayer : it->dstOffset.z,
            it->extent.width,
            dstType == IGPUImage::ET_1D ? it->dstSubresource.layerCount : it->extent.height,
            dstType == IGPUImage::ET_2D ? it->dstSubresource.layerCount : it->extent.depth))
        {
            return false;
        }
    }

    return true;
}

bool COpenGLCommandBuffer::copyBufferToImage_impl(const buffer_t* srcBuffer, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions)
{
    if (!dstImage->validateCopies(pRegions, pRegions + regionCount, srcBuffer))
        return false;

    const auto params = dstImage->getCreationParameters();
    const auto type = params.type;
    const auto format = params.format;
    const bool compressed = asset::isBlockCompressionFormat(format);
    auto dstImageGL = static_cast<COpenGLImage*>(dstImage);
    GLuint dst = dstImageGL->getOpenGLName();
    GLenum glfmt, gltype;
    getOpenGLFormatAndParametersFromColorFormat(m_features, format, glfmt, gltype);

    const auto bpp = asset::getBytesPerPixel(format);
    const auto blockDims = asset::getBlockDimensions(format);
    const auto blockByteSize = asset::getTexelOrBlockBytesize(format);

    m_stateCache.nextState.pixelUnpack.buffer = core::smart_refctd_ptr<const COpenGLBuffer>(static_cast<const COpenGLBuffer*>(srcBuffer));

    bool anyFailed = false;
    for (auto it = pRegions; it != pRegions + regionCount; it++)
    {
        if (it->bufferOffset != core::alignUp(it->bufferOffset, blockByteSize))
        {
            assert(!"bufferOffset should be aligned to block/texel byte size.");
            continue;
        }

        uint32_t pitch = ((it->bufferRowLength ? it->bufferRowLength : it->imageExtent.width) * bpp).getIntegerApprox();
        int32_t alignment = 0x1 << core::min(core::min<uint32_t>(core::findLSB(it->bufferOffset), core::findLSB(pitch)), 3u);

        m_stateCache.nextState.pixelUnpack.alignment = alignment;
        m_stateCache.nextState.pixelUnpack.rowLength = it->bufferRowLength;
        m_stateCache.nextState.pixelUnpack.imgHeight = it->bufferImageHeight;

        bool success = true;
        if (compressed)
        {
            m_stateCache.nextState.pixelUnpack.BCwidth = blockDims[0];
            m_stateCache.nextState.pixelUnpack.BCheight = blockDims[1];
            m_stateCache.nextState.pixelUnpack.BCdepth = blockDims[2];

            if (!m_stateCache.flushStateGraphics(SOpenGLContextIndependentCache::GSB_PIXEL_PACK_UNPACK, m_cmdpool.get(), m_GLSegmentListHeadItr, m_GLSegmentListTail, getAPIType(), m_features))
                success = false;

            uint32_t imageSize = pitch;
            switch (type)
            {
            case IGPUImage::ET_1D:
            {
                imageSize *= it->imageSubresource.layerCount;

                if (!m_cmdpool->emplace<COpenGLCommandPool::CCompressedTextureSubImage2DCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail,
                    dst,
                    GL_TEXTURE_1D_ARRAY,
                    it->imageSubresource.mipLevel,
                    it->imageOffset.x,
                    it->imageSubresource.baseArrayLayer,
                    it->imageExtent.width,
                    it->imageSubresource.layerCount,
                    dstImageGL->getOpenGLSizedFormat(),
                    imageSize,
                    reinterpret_cast<const void*>(it->bufferOffset)))
                    success = false;
            } break;

            case IGPUImage::ET_2D:
            {
                imageSize *= (it->bufferImageHeight ? it->bufferImageHeight : it->imageExtent.height);
                imageSize *= it->imageSubresource.layerCount;

                if (!m_cmdpool->emplace<COpenGLCommandPool::CCompressedTextureSubImage3DCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail,
                    dst,
                    GL_TEXTURE_2D_ARRAY,
                    it->imageSubresource.mipLevel,
                    it->imageOffset.x,
                    it->imageOffset.y,
                    it->imageSubresource.baseArrayLayer,
                    it->imageExtent.width,
                    it->imageExtent.height,
                    it->imageSubresource.layerCount,
                    dstImageGL->getOpenGLSizedFormat(),
                    imageSize,
                    reinterpret_cast<const void*>(it->bufferOffset)))
                    success = false;
            } break;

            case IGPUImage::ET_3D:
            {
                imageSize *= (it->bufferImageHeight ? it->bufferImageHeight : it->imageExtent.height);
                imageSize *= it->imageExtent.depth;

                if (!m_cmdpool->emplace<COpenGLCommandPool::CCompressedTextureSubImage3DCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail,
                    dst,
                    GL_TEXTURE_3D,
                    it->imageSubresource.mipLevel,
                    it->imageOffset.x,
                    it->imageOffset.y,
                    it->imageOffset.z,
                    it->imageExtent.width,
                    it->imageExtent.height,
                    it->imageExtent.depth,
                    dstImageGL->getOpenGLSizedFormat(),
                    imageSize,
                    reinterpret_cast<const void*>(it->bufferOffset)))
                    success = false;
            } break;

            default:
            {
                assert(!"Invalid code path");
                success = false;
            } break;
            }
        }
        else
        {
            if (!m_stateCache.flushStateGraphics(SOpenGLContextIndependentCache::GSB_PIXEL_PACK_UNPACK, m_cmdpool.get(), m_GLSegmentListHeadItr, m_GLSegmentListTail, getAPIType(), m_features))
                success = false;

            switch (type)
            {
            case IGPUImage::ET_1D:
            {
                if (!m_cmdpool->emplace<COpenGLCommandPool::CTextureSubImage2DCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail,
                    dst,
                    GL_TEXTURE_1D_ARRAY,
                    it->imageSubresource.mipLevel,
                    it->imageOffset.x,
                    it->imageSubresource.baseArrayLayer,
                    it->imageExtent.width,
                    it->imageSubresource.layerCount,
                    glfmt,
                    gltype,
                    reinterpret_cast<const void*>(it->bufferOffset)))
                    success = false;
            } break;

            case IGPUImage::ET_2D:
            {
                if (!m_cmdpool->emplace<COpenGLCommandPool::CTextureSubImage3DCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail,
                    dst,
                    GL_TEXTURE_2D_ARRAY,
                    it->imageSubresource.mipLevel,
                    it->imageOffset.x,
                    it->imageOffset.y,
                    it->imageSubresource.baseArrayLayer,
                    it->imageExtent.width,
                    it->imageExtent.height,
                    it->imageSubresource.layerCount,
                    glfmt,
                    gltype,
                    reinterpret_cast<const void*>(it->bufferOffset)))
                    success = false;
            } break;

            case IGPUImage::ET_3D:
            {
                if (!m_cmdpool->emplace<COpenGLCommandPool::CTextureSubImage3DCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail,
                    dst,
                    GL_TEXTURE_3D,
                    it->imageSubresource.mipLevel,
                    it->imageOffset.x,
                    it->imageOffset.y,
                    it->imageOffset.z,
                    it->imageExtent.width,
                    it->imageExtent.height,
                    it->imageExtent.depth,
                    glfmt,
                    gltype,
                    reinterpret_cast<const void*>(it->bufferOffset)))
                    success = false;
            } break;

            default:
            {
                assert(!"Invalid code path");
                success = false;
            } break;
            }
        }

        if (!success)
            anyFailed = true;
    }

    return !anyFailed;
}

bool COpenGLCommandBuffer::copyImageToBuffer_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, buffer_t* dstBuffer, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions)
{
    if (!srcImage->validateCopies(pRegions, pRegions + regionCount, dstBuffer))
        return false;

    const auto params = srcImage->getCreationParameters();
    const auto type = params.type;
    const auto format = params.format;
    const uint32_t fmtBytesize = nbl::asset::getTexelOrBlockBytesize(format);
    const bool compressed = asset::isBlockCompressionFormat(format);
    const auto* glimg = static_cast<const COpenGLImage*>(srcImage);
    GLuint src = glimg->getOpenGLName();
    GLenum glfmt, gltype;
    getOpenGLFormatAndParametersFromColorFormat(m_features, format, glfmt, gltype);

    const auto texelBlockInfo = asset::TexelBlockInfo(format);
    const auto bpp = asset::getBytesPerPixel(format);
    const auto blockDims = texelBlockInfo.getDimension();
    const auto blockByteSize = texelBlockInfo.getBlockByteSize();

    const bool usingGetTexSubImage = (m_features->Version >= 450 || m_features->FeatureAvailable[m_features->EOpenGLFeatures::NBL_ARB_get_texture_sub_image]);

    m_stateCache.nextState.pixelPack.buffer = core::smart_refctd_ptr<const COpenGLBuffer>(static_cast<COpenGLBuffer*>(dstBuffer));

    bool anyFailed = false;
    for (auto it = pRegions; it != pRegions + regionCount; it++)
    {
        if (it->bufferOffset != core::alignUp(it->bufferOffset, blockByteSize))
        {
            m_logger.log("bufferOffset should be aligned to block/texel byte size.", system::ILogger::ELL_ERROR);
            assert(false);
            continue;
        }

        const auto imageExtent = core::vector3du32_SIMD(it->imageExtent.width, it->imageExtent.height, it->imageExtent.depth);
        const auto imageExtentInBlocks = texelBlockInfo.convertTexelsToBlocks(imageExtent);
        const auto imageExtentBlockStridesInBytes = texelBlockInfo.convert3DBlockStridesTo1DByteStrides(imageExtentInBlocks);
        const uint32_t eachLayerNeededMemory = imageExtentBlockStridesInBytes[3];  // = blockByteSize * imageExtentInBlocks.x * imageExtentInBlocks.y * imageExtentInBlocks.z

        uint32_t pitch = ((it->bufferRowLength ? it->bufferRowLength : it->imageExtent.width) * bpp).getIntegerApprox();
        int32_t alignment = 0x1 << core::min(core::max(core::findLSB(it->bufferOffset), core::findLSB(pitch)), 3u);

        m_stateCache.nextState.pixelPack.alignment = alignment;
        m_stateCache.nextState.pixelPack.rowLength = it->bufferRowLength;
        m_stateCache.nextState.pixelPack.imgHeight = it->bufferImageHeight;

        auto yStart = type == IGPUImage::ET_1D ? it->imageSubresource.baseArrayLayer : it->imageOffset.y;
        auto yRange = type == IGPUImage::ET_1D ? it->imageSubresource.layerCount : it->imageExtent.height;
        auto zStart = type == IGPUImage::ET_2D ? it->imageSubresource.baseArrayLayer : it->imageOffset.z;
        auto zRange = type == IGPUImage::ET_2D ? it->imageSubresource.layerCount : it->imageExtent.depth;

        bool success = true;
        if (usingGetTexSubImage)
        {
            if (compressed)
            {
                m_stateCache.nextState.pixelPack.BCwidth = blockDims[0];
                m_stateCache.nextState.pixelPack.BCheight = blockDims[1];
                m_stateCache.nextState.pixelPack.BCdepth = blockDims[2];
            }

            if (!m_stateCache.flushStateGraphics(SOpenGLContextIndependentCache::GSB_PIXEL_PACK_UNPACK, m_cmdpool.get(), m_GLSegmentListHeadItr, m_GLSegmentListTail, getAPIType(), m_features))
                success = false;

            if (compressed)
            {
                if (!m_cmdpool->emplace<COpenGLCommandPool::CGetCompressedTextureSubImageCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail,
                    src,
                    it->imageSubresource.mipLevel,
                    it->imageOffset.x,
                    yStart,
                    zStart,
                    it->imageExtent.width,
                    yRange,
                    zRange,
                    dstBuffer->getSize() - it->bufferOffset,
                    it->bufferOffset))
                    success = false;
            }
            else
            {
                if (!m_cmdpool->emplace<COpenGLCommandPool::CGetTextureSubImageCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail,
                    src,
                    it->imageSubresource.mipLevel,
                    it->imageOffset.x,
                    yStart,
                    zStart,
                    it->imageExtent.width,
                    yRange,
                    zRange,
                    glfmt,
                    gltype,
                    dstBuffer->getSize() - it->bufferOffset,
                    it->bufferOffset))
                    success = false;
            }
        }
        else
        {
            success = m_stateCache.flushStateGraphics(SOpenGLContextIndependentCache::GSB_PIXEL_PACK_UNPACK, m_cmdpool.get(), m_GLSegmentListHeadItr, m_GLSegmentListTail, getAPIType(), m_features);

            const size_t bytesPerLayer = eachLayerNeededMemory;
            for (uint32_t z = 0u; z < zRange; ++z)
            {
                size_t bufOffset = it->bufferOffset + z * bytesPerLayer;
                bufOffset = core::alignUp(bufOffset, alignment); // ??? am i doing it right?
                if (!m_cmdpool->emplace<COpenGLCommandPool::CReadPixelsCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail,
                    glimg,
                    it->imageSubresource.mipLevel,
                    it->imageSubresource.baseArrayLayer + z,
                    it->imageOffset.x,
                    yStart,
                    it->imageExtent.width,
                    yRange,
                    glfmt,
                    gltype,
                    bufOffset))
                    success = false;
            }
        }

        if (!success)
            anyFailed = true;
    }

    return !anyFailed;
}

bool COpenGLCommandBuffer::blitImage_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageBlit* pRegions, asset::ISampler::E_TEXTURE_FILTER filter)
{
    const auto* gl_srcImage = static_cast<const COpenGLImage*>(srcImage);
    const auto* gl_dstImage = static_cast<const COpenGLImage*>(dstImage);
    for (uint32_t i = 0u; i < regionCount; ++i)
    {
        auto& info = pRegions[i];

        assert(info.dstSubresource.layerCount == info.srcSubresource.layerCount);

        for (uint32_t l = 0u; l < info.dstSubresource.layerCount; ++l)
        {
            if (!m_cmdpool->emplace<COpenGLCommandPool::CBlitNamedFramebufferCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, gl_srcImage, gl_dstImage, info.srcSubresource.mipLevel, info.dstSubresource.mipLevel, info.srcSubresource.baseArrayLayer+l, info.dstSubresource.baseArrayLayer+l, info.srcOffsets, info.dstOffsets, filter))
                return false;
        }
    }

    return true;
}

bool COpenGLCommandBuffer::resolveImage_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageResolve* pRegions)
{
    for (uint32_t i = 0u; i < regionCount; ++i)
    {
        auto& info = pRegions[i];

        assert(info.dstSubresource.layerCount == info.srcSubresource.layerCount);

        asset::VkOffset3D srcoffsets[2]{ info.srcOffset,info.srcOffset };
        srcoffsets[1].x += info.extent.width;
        srcoffsets[1].y += info.extent.height;
        srcoffsets[1].z += info.extent.depth;
        asset::VkOffset3D dstoffsets[2]{ info.dstOffset,info.dstOffset };
        dstoffsets[1].x += info.extent.width;
        dstoffsets[1].y += info.extent.height;
        dstoffsets[1].z += info.extent.depth;

        for (uint32_t l = 0u; l < info.dstSubresource.layerCount; ++l)
        {
            if (!m_cmdpool->emplace<COpenGLCommandPool::CBlitNamedFramebufferCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, static_cast<const COpenGLImage*>(srcImage), static_cast<const COpenGLImage*>(dstImage), info.srcSubresource.mipLevel, info.dstSubresource.mipLevel, info.srcSubresource.baseArrayLayer + l, info.dstSubresource.baseArrayLayer + l, srcoffsets, dstoffsets, asset::ISampler::ETF_NEAREST))
                return false;
        }
    }

    return true;
}

bool COpenGLCommandBuffer::drawIndirect_impl(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride)
{
    m_stateCache.nextState.vertexInputParams.indirectDrawBuf = static_cast<const COpenGLBuffer*>(buffer);
    m_stateCache.nextState.vertexInputParams.parameterBuf = nullptr;

    if (!m_stateCache.flushStateGraphics(SOpenGLContextIndependentCache::GSB_ALL, m_cmdpool.get(), m_GLSegmentListHeadItr, m_GLSegmentListTail, getAPIType(), m_features))
        return false;

    const asset::E_PRIMITIVE_TOPOLOGY primType = m_stateCache.currentState.pipeline.graphics.pipeline->getRenderpassIndependentPipeline()->getPrimitiveAssemblyParams().primitiveType;
    GLenum glpt = getGLprimitiveType(primType);

    if (!m_cmdpool->emplace<COpenGLCommandPool::CMultiDrawArraysIndirectCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, glpt, (GLuint64)offset, drawCount, stride))
        return false;

    return true;
}

bool COpenGLCommandBuffer::drawIndexedIndirect_impl(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride)
{
    m_stateCache.nextState.vertexInputParams.indirectDrawBuf = static_cast<const COpenGLBuffer*>(buffer);
    m_stateCache.nextState.vertexInputParams.parameterBuf = nullptr;

    if (!m_stateCache.flushStateGraphics(SOpenGLContextIndependentCache::GSB_ALL, m_cmdpool.get(), m_GLSegmentListHeadItr, m_GLSegmentListTail, getAPIType(), m_features))
        return false;

    GLenum idxType = GL_INVALID_ENUM;
    switch (m_stateCache.currentState.vertexInputParams.vaoval.idxType)
    {
    case asset::EIT_16BIT:
        idxType = GL_UNSIGNED_SHORT;
        break;
    case asset::EIT_32BIT:
        idxType = GL_UNSIGNED_INT;
        break;
    default:
        assert(!"Invalid code path.");
        break;
    }

    const asset::E_PRIMITIVE_TOPOLOGY primType = m_stateCache.currentState.pipeline.graphics.pipeline->getRenderpassIndependentPipeline()->getPrimitiveAssemblyParams().primitiveType;
    GLenum glpt = getGLprimitiveType(primType);

    if (!m_cmdpool->emplace<COpenGLCommandPool::CMultiDrawElementsIndirectCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, glpt, idxType, (GLuint64)offset, drawCount, stride))
        return false;

    return true;
}

bool COpenGLCommandBuffer::drawIndirectCount_impl(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride)
{
    m_stateCache.nextState.vertexInputParams.indirectDrawBuf = static_cast<const COpenGLBuffer*>(buffer);
    m_stateCache.nextState.vertexInputParams.parameterBuf = static_cast<const COpenGLBuffer*>(countBuffer);

    if (!m_stateCache.flushStateGraphics(SOpenGLContextIndependentCache::GSB_ALL, m_cmdpool.get(), m_GLSegmentListHeadItr, m_GLSegmentListTail, getAPIType(), m_features))
        return false;

    const asset::E_PRIMITIVE_TOPOLOGY primType = m_stateCache.currentState.pipeline.graphics.pipeline->getRenderpassIndependentPipeline()->getPrimitiveAssemblyParams().primitiveType;
    GLenum glpt = getGLprimitiveType(primType);

    if (!m_cmdpool->emplace<COpenGLCommandPool::CMultiDrawArraysIndirectCountCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, glpt, (GLuint64)offset, countBufferOffset, maxDrawCount, stride))
        return false;

    return true;
}

bool COpenGLCommandBuffer::drawIndexedIndirectCount_impl(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride)
{
    m_stateCache.nextState.vertexInputParams.indirectDrawBuf = static_cast<const COpenGLBuffer*>(buffer);
    m_stateCache.nextState.vertexInputParams.parameterBuf = static_cast<const COpenGLBuffer*>(countBuffer);

    if (!m_stateCache.flushStateGraphics(SOpenGLContextIndependentCache::GSB_ALL, m_cmdpool.get(), m_GLSegmentListHeadItr, m_GLSegmentListTail, getAPIType(), m_features))
        return false;

    GLenum idxType = GL_INVALID_ENUM;
    switch (m_stateCache.currentState.vertexInputParams.vaoval.idxType)
    {
    case asset::EIT_16BIT:
        idxType = GL_UNSIGNED_SHORT;
        break;
    case asset::EIT_32BIT:
        idxType = GL_UNSIGNED_INT;
        break;
    default:
        break;
    }

    const asset::E_PRIMITIVE_TOPOLOGY primType = m_stateCache.currentState.pipeline.graphics.pipeline->getRenderpassIndependentPipeline()->getPrimitiveAssemblyParams().primitiveType;
    GLenum glpt = getGLprimitiveType(primType);

    if (!m_cmdpool->emplace<COpenGLCommandPool::CMultiDrawElementsIndirectCountCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, glpt, idxType, (GLuint64)offset, countBufferOffset, maxDrawCount, stride))
        return false;

    return true;
}

bool COpenGLCommandBuffer::setViewport(uint32_t firstViewport, uint32_t viewportCount, const asset::SViewport* pViewports)
{
    if (viewportCount == 0u)
        return false;

    if (firstViewport >= SOpenGLState::MAX_VIEWPORT_COUNT)
        return false;

    uint32_t count = std::min(viewportCount, SOpenGLState::MAX_VIEWPORT_COUNT);
    if (firstViewport + count > SOpenGLState::MAX_VIEWPORT_COUNT)
        count = SOpenGLState::MAX_VIEWPORT_COUNT - firstViewport;

    uint32_t first = firstViewport;
    for (uint32_t i = 0u; i < count; ++i)
    {
        auto& vp = m_stateCache.nextState.rasterParams.viewport[first + i];
        auto& vpd = m_stateCache.nextState.rasterParams.viewport_depth[first + i];

        vp.x = pViewports[i].x;
        vp.y = pViewports[i].y;
        vp.width = pViewports[i].width;
        vp.height = pViewports[i].height;
        vpd.minDepth = pViewports[i].minDepth;
        vpd.maxDepth = pViewports[i].maxDepth;
    }

    return true;
}

bool COpenGLCommandBuffer::setLineWidth(float lineWidth)
{
    m_stateCache.nextState.rasterParams.lineWidth = lineWidth;

    return true;
}

bool COpenGLCommandBuffer::setDepthBias(float depthBiasConstantFactor, float depthBiasClamp, float depthBiasSlopeFactor)
{
    // TODO what about c.depthBiasClamp
    m_stateCache.nextState.rasterParams.polygonOffset.factor = depthBiasSlopeFactor;
    m_stateCache.nextState.rasterParams.polygonOffset.units = depthBiasConstantFactor;
    return true;
}

bool COpenGLCommandBuffer::setBlendConstants(const float blendConstants[4])
{
    // TODO, cant see such thing in opengl
    return true;
}

bool COpenGLCommandBuffer::executeCommands_impl(uint32_t count, IGPUCommandBuffer* const* const cmdbufs)
{
    if (!m_cmdpool->emplace<COpenGLCommandPool::CExecuteCommandsCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, count, cmdbufs))
        return false;

    return true;
}

bool COpenGLCommandBuffer::regenerateMipmaps_impl(image_t* img, uint32_t lastReadyMip, asset::IImage::E_ASPECT_FLAGS aspect)
{
    auto* glimg = static_cast<COpenGLImage*>(img);
    if (!m_cmdpool->emplace<COpenGLCommandPool::CGenerateTextureMipmapCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, glimg->getOpenGLName(), glimg->getOpenGLTarget()))
        return false;

    return true;
}

}