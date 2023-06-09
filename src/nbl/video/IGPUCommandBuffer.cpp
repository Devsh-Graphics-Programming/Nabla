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
    m_resetCheckedStamp = m_cmdpool->getResetCounter();

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

    if (inheritanceInfo)
    {
        if (!inheritanceInfo->renderpass || inheritanceInfo->renderpass->getAPIType() != getAPIType() || !inheritanceInfo->renderpass->isCompatibleDevicewise(this))
            return false;

        if (inheritanceInfo->framebuffer && (inheritanceInfo->framebuffer->getAPIType() != getAPIType() || !inheritanceInfo->framebuffer->isCompatibleDevicewise(this)))
            return false;

        if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBeginCmd>(m_commandList, core::smart_refctd_ptr<const IGPURenderpass>(inheritanceInfo->renderpass.get()), core::smart_refctd_ptr<const IGPUFramebuffer>(inheritanceInfo->framebuffer.get())))
            return false;
    }

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
    if (!checkStateBeforeRecording())
        return false;

    m_state = ES_EXECUTABLE;
    return end_impl();
}

bool IGPUCommandBuffer::bindIndexBuffer(const buffer_t* buffer, size_t offset, asset::E_INDEX_TYPE indexType)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!buffer || (buffer->getAPIType() != getAPIType()))
        return false;

    if (!this->isCompatibleDevicewise(buffer))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBindIndexBufferCmd>(m_commandList, core::smart_refctd_ptr<const IGPUBuffer>(buffer)))
        return false;

    bindIndexBuffer_impl(buffer, offset, indexType);

    return true;
}

bool IGPUCommandBuffer::drawIndirect(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!buffer || (buffer->getAPIType() != getAPIType()))
        return false;

    if (drawCount == 0u)
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CDrawIndirectCmd>(m_commandList, core::smart_refctd_ptr<const IGPUBuffer>(buffer)))
        return false;

    return drawIndirect_impl(buffer, offset, drawCount, stride);
}

bool IGPUCommandBuffer::drawIndexedIndirect(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!buffer || buffer->getAPIType() != getAPIType())
        return false;

    if (drawCount == 0u)
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CDrawIndexedIndirectCmd>(m_commandList, core::smart_refctd_ptr<const buffer_t>(buffer)))
        return false;

    drawIndexedIndirect_impl(buffer, offset, drawCount, stride);

    return true;
}

bool IGPUCommandBuffer::drawIndirectCount(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!buffer || buffer->getAPIType() != getAPIType())
        return false;

    if (!countBuffer || countBuffer->getAPIType() != getAPIType())
        return false;

    if (maxDrawCount == 0u)
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CDrawIndirectCountCmd>(m_commandList, core::smart_refctd_ptr<const buffer_t>(buffer), core::smart_refctd_ptr<const buffer_t>(countBuffer)))
        return false;

    return drawIndirectCount_impl(buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
}

bool IGPUCommandBuffer::drawIndexedIndirectCount(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!buffer || buffer->getAPIType() != getAPIType())
        return false;

    if (!countBuffer || countBuffer->getAPIType() != getAPIType())
        return false;

    if (maxDrawCount == 0u)
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CDrawIndexedIndirectCountCmd>(m_commandList, core::smart_refctd_ptr<const buffer_t>(buffer), core::smart_refctd_ptr<const buffer_t>(countBuffer)))
        return false;

    return drawIndexedIndirectCount_impl(buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);}

bool IGPUCommandBuffer::beginRenderPass(const SRenderpassBeginInfo* pRenderPassBegin, asset::E_SUBPASS_CONTENTS content)
{
    if (!checkStateBeforeRecording())
        return false;

    const auto apiType = getAPIType();
    if ((apiType != pRenderPassBegin->renderpass->getAPIType()) || (apiType != pRenderPassBegin->framebuffer->getAPIType()))
        return false;

    if (!this->isCompatibleDevicewise(pRenderPassBegin->framebuffer.get()))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBeginRenderPassCmd>(m_commandList, core::smart_refctd_ptr<const IGPURenderpass>(pRenderPassBegin->renderpass), core::smart_refctd_ptr<const IGPUFramebuffer>(pRenderPassBegin->framebuffer)))
        return false;

    return beginRenderPass_impl(pRenderPassBegin, content);
}

bool IGPUCommandBuffer::pipelineBarrier(core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> srcStageMask, core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> dstStageMask,
    core::bitflag<asset::E_DEPENDENCY_FLAGS> dependencyFlags,
    uint32_t memoryBarrierCount, const asset::SMemoryBarrier* pMemoryBarriers,
    uint32_t bufferMemoryBarrierCount, const SBufferMemoryBarrier* pBufferMemoryBarriers,
    uint32_t imageMemoryBarrierCount, const SImageMemoryBarrier* pImageMemoryBarriers)
{
    if (!checkStateBeforeRecording())
        return false;

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

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CPipelineBarrierCmd>(m_commandList, bufferMemoryBarrierCount, bufferResources, imageMemoryBarrierCount, imageResources))
        return false;

    pipelineBarrier_impl(srcStageMask, dstStageMask, dependencyFlags, memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);

    return true;
}

bool IGPUCommandBuffer::bindDescriptorSets(asset::E_PIPELINE_BIND_POINT pipelineBindPoint, const pipeline_layout_t* layout, uint32_t firstSet, const uint32_t descriptorSetCount,
    const descriptor_set_t* const* const pDescriptorSets, const uint32_t dynamicOffsetCount, const uint32_t* dynamicOffsets)
{
    if (!checkStateBeforeRecording())
        return false;

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

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBindDescriptorSetsCmd>(m_commandList, core::smart_refctd_ptr<const IGPUPipelineLayout>(layout), descriptorSetCount, pDescriptorSets))
        return false;

    for (uint32_t i = 0u; i < descriptorSetCount; ++i)
    {
        if (pDescriptorSets[i] && !pDescriptorSets[i]->getLayout()->canUpdateAfterBind())
        {
            const auto currentVersion = pDescriptorSets[i]->getVersion();

            auto found = m_boundDescriptorSetsRecord.find(pDescriptorSets[i]);

            if (found != m_boundDescriptorSetsRecord.end())
            {
                if (found->second != currentVersion)
                {
                    const char* debugName = pDescriptorSets[i]->getDebugName();
                    if (debugName)
                        m_logger.log("Descriptor set (%s, %p) was modified between two recorded bind commands since the last command buffer's beginning.", system::ILogger::ELL_ERROR, debugName, pDescriptorSets[i]);
                    else
                        m_logger.log("Descriptor set (%p)  was modified between two recorded bind commands since the last command buffer's beginning.", system::ILogger::ELL_ERROR, pDescriptorSets[i]);

                    m_state = ES_INVALID;
                    return false;
                }
            }
            else
            {
                m_boundDescriptorSetsRecord.insert({ pDescriptorSets[i], currentVersion });
            }
        }
    }

    return bindDescriptorSets_impl(pipelineBindPoint, layout, firstSet, descriptorSetCount, pDescriptorSets, dynamicOffsetCount, dynamicOffsets);
}

bool IGPUCommandBuffer::bindComputePipeline(const compute_pipeline_t* pipeline)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!this->isCompatibleDevicewise(pipeline))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBindComputePipelineCmd>(m_commandList, core::smart_refctd_ptr<const IGPUComputePipeline>(pipeline)))
        return false;

    bindComputePipeline_impl(pipeline);

    return true;
}

bool IGPUCommandBuffer::updateBuffer(buffer_t* dstBuffer, size_t dstOffset, size_t dataSize, const void* pData)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!validate_updateBuffer(dstBuffer, dstOffset, dataSize, pData))
    {
        m_logger.log("Invalid arguments see `IGPUCommandBuffer::validate_updateBuffer`.", system::ILogger::ELL_ERROR);
        return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CUpdateBufferCmd>(m_commandList, core::smart_refctd_ptr<const IGPUBuffer>(dstBuffer)))
        return false;

    return updateBuffer_impl(dstBuffer, dstOffset, dataSize, pData);
}

static void getResourcesFromBuildGeometryInfos(const core::SRange<video::IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, core::vector<core::smart_refctd_ptr<const IGPUAccelerationStructure>>& accelerationStructures, core::vector<core::smart_refctd_ptr<const IGPUBuffer>>& buffers)
{
    const size_t infoCount = pInfos.size();
    IGPUAccelerationStructure::DeviceBuildGeometryInfo* infos = pInfos.begin();

    static constexpr size_t MaxGeometryPerBuildInfoCount = 64;

    // * 2 because of info.srcAS + info.dstAS 
    accelerationStructures.reserve(infoCount * 2);

    // + 1 because of info.scratchAddr.buffer
    // * 3 because of worst-case all triangle data ( vertexData + indexData + transformData+
    buffers.reserve((1 + MaxGeometryPerBuildInfoCount * 3) * infoCount);

    for (uint32_t i = 0; i < infoCount; ++i)
    {
        accelerationStructures.push_back(core::smart_refctd_ptr<const IGPUAccelerationStructure>(infos[i].srcAS));
        accelerationStructures.push_back(core::smart_refctd_ptr<const IGPUAccelerationStructure>(infos[i].dstAS));
        buffers.push_back(infos[i].scratchAddr.buffer);

        if (!infos[i].geometries.empty())
        {
            const auto geomCount = infos[i].geometries.size();
            assert(geomCount > 0);
            assert(geomCount <= MaxGeometryPerBuildInfoCount);

            auto* geoms = infos[i].geometries.begin();
            for (uint32_t g = 0; g < geomCount; ++g)
            {
                auto const& geometry = geoms[g];

                if (IGPUAccelerationStructure::E_GEOM_TYPE::EGT_TRIANGLES == geometry.type)
                {
                    auto const& triangles = geometry.data.triangles;
                    if (triangles.vertexData.isValid())
                        buffers.push_back(triangles.vertexData.buffer);
                    if (triangles.indexData.isValid())
                        buffers.push_back(triangles.indexData.buffer);
                    if (triangles.transformData.isValid())
                        buffers.push_back(triangles.transformData.buffer);
                }
                else if (IGPUAccelerationStructure::E_GEOM_TYPE::EGT_AABBS == geometry.type)
                {
                    const auto& aabbs = geometry.data.aabbs;
                    if (aabbs.data.isValid())
                        buffers.push_back(aabbs.data.buffer);
                }
                else if (IGPUAccelerationStructure::E_GEOM_TYPE::EGT_INSTANCES == geometry.type)
                {
                    const auto& instances = geometry.data.instances;
                    if (instances.data.isValid())
                        buffers.push_back(instances.data.buffer);
                }
            }
        }
    }
}

bool IGPUCommandBuffer::buildAccelerationStructures(const core::SRange<video::IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, video::IGPUAccelerationStructure::BuildRangeInfo* const* ppBuildRangeInfos)
{
    if (!checkStateBeforeRecording())
        return false;

    if (pInfos.empty())
        return false;

    core::vector<core::smart_refctd_ptr<const IGPUAccelerationStructure>> accelerationStructures;
    core::vector<core::smart_refctd_ptr<const IGPUBuffer>> buffers;
    getResourcesFromBuildGeometryInfos(pInfos, accelerationStructures, buffers);

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBuildAccelerationStructuresCmd>(m_commandList, accelerationStructures.size(), accelerationStructures.data(), buffers.size(), buffers.data()))
        return false;
    
    return buildAccelerationStructures_impl(pInfos, ppBuildRangeInfos);
}

bool IGPUCommandBuffer::buildAccelerationStructuresIndirect(const core::SRange<video::IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, const core::SRange<video::IGPUAccelerationStructure::DeviceAddressType>& pIndirectDeviceAddresses, const uint32_t* pIndirectStrides, const uint32_t* const* ppMaxPrimitiveCounts)
{
    if (!checkStateBeforeRecording())
        return false;

    if (pInfos.empty())
        return false;

    core::vector<core::smart_refctd_ptr<const IGPUAccelerationStructure>> accelerationStructures;
    core::vector<core::smart_refctd_ptr<const IGPUBuffer>> buffers;
    getResourcesFromBuildGeometryInfos(pInfos, accelerationStructures, buffers);

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBuildAccelerationStructuresCmd>(m_commandList, accelerationStructures.size(), accelerationStructures.data(), buffers.size(), buffers.data()))
        return false;

    return buildAccelerationStructuresIndirect_impl(pInfos, pIndirectDeviceAddresses, pIndirectStrides, ppMaxPrimitiveCounts);
}

bool IGPUCommandBuffer::copyAccelerationStructure(const video::IGPUAccelerationStructure::CopyInfo& copyInfo)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!copyInfo.src || copyInfo.src->getAPIType() != getAPIType())
        return false;

    if (!copyInfo.dst || copyInfo.dst->getAPIType() != getAPIType())
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyAccelerationStructureCmd>(m_commandList, core::smart_refctd_ptr<const IGPUAccelerationStructure>(copyInfo.src), core::smart_refctd_ptr<const IGPUAccelerationStructure>(copyInfo.dst)))
        return false;

    return copyAccelerationStructure_impl(copyInfo);
}

bool IGPUCommandBuffer::copyAccelerationStructureToMemory(const video::IGPUAccelerationStructure::DeviceCopyToMemoryInfo& copyInfo)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!copyInfo.src || copyInfo.src->getAPIType() != getAPIType())
        return false;

    if (!copyInfo.dst.buffer || copyInfo.dst.buffer->getAPIType() != getAPIType())
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyAccelerationStructureToOrFromMemoryCmd>(m_commandList, core::smart_refctd_ptr<const IGPUAccelerationStructure>(copyInfo.src), core::smart_refctd_ptr<const IGPUBuffer>(copyInfo.dst.buffer)))
        return false;

    return copyAccelerationStructureToMemory_impl(copyInfo);
}

bool IGPUCommandBuffer::copyAccelerationStructureFromMemory(const video::IGPUAccelerationStructure::DeviceCopyFromMemoryInfo& copyInfo)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!copyInfo.src.buffer || copyInfo.src.buffer->getAPIType() != getAPIType())
        return false;

    if (!copyInfo.dst || copyInfo.dst->getAPIType() != getAPIType())
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyAccelerationStructureToOrFromMemoryCmd>(m_commandList, core::smart_refctd_ptr<const IGPUAccelerationStructure>(copyInfo.dst), core::smart_refctd_ptr<const IGPUBuffer>(copyInfo.src.buffer)))
        return false;

    return copyAccelerationStructureFromMemory_impl(copyInfo);
}

bool IGPUCommandBuffer::resetQueryPool(video::IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!queryPool || !this->isCompatibleDevicewise(queryPool))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CResetQueryPoolCmd>(m_commandList, core::smart_refctd_ptr<const IQueryPool>(queryPool)))
        return false;

    return resetQueryPool_impl(queryPool, firstQuery, queryCount);
}

bool IGPUCommandBuffer::writeTimestamp(asset::E_PIPELINE_STAGE_FLAGS pipelineStage, video::IQueryPool* queryPool, uint32_t query)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!queryPool || !this->isCompatibleDevicewise(queryPool))
        return false;

    assert(core::isPoT(static_cast<uint32_t>(pipelineStage))); // should only be 1 stage (1 bit set)

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CWriteTimestampCmd>(m_commandList, core::smart_refctd_ptr<const IQueryPool>(queryPool)))
        return false;

    return writeTimestamp_impl(pipelineStage, queryPool, query);
}

bool IGPUCommandBuffer::writeAccelerationStructureProperties(const core::SRange<video::IGPUAccelerationStructure>& pAccelerationStructures, video::IQueryPool::E_QUERY_TYPE queryType, video::IQueryPool* queryPool, uint32_t firstQuery)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!queryPool || pAccelerationStructures.empty())
        return false;

    const uint32_t asCount = static_cast<uint32_t>(pAccelerationStructures.size());
    // TODO: Use Better Containers
    static constexpr size_t MaxAccelerationStructureCount = 128;
    assert(asCount <= MaxAccelerationStructureCount);

    const IGPUAccelerationStructure* accelerationStructures[MaxAccelerationStructureCount] = { nullptr };
    for (auto i = 0; i < asCount; ++i)
        accelerationStructures[i] = &pAccelerationStructures.begin()[i];

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CWriteAccelerationStructurePropertiesCmd>(m_commandList, queryPool, asCount, accelerationStructures))
        return false;

    return writeAccelerationStructureProperties_impl(pAccelerationStructures, queryType, queryPool, firstQuery);
}

bool IGPUCommandBuffer::beginQuery(video::IQueryPool* queryPool, uint32_t query, core::bitflag<video::IQueryPool::E_QUERY_CONTROL_FLAGS> flags)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!queryPool || !this->isCompatibleDevicewise(queryPool))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBeginQueryCmd>(m_commandList, core::smart_refctd_ptr<const IQueryPool>(queryPool)))
        return false;

    return beginQuery_impl(queryPool, query, flags);
}

bool IGPUCommandBuffer::endQuery(video::IQueryPool* queryPool, uint32_t query)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!queryPool || !this->isCompatibleDevicewise(queryPool))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CEndQueryCmd>(m_commandList, core::smart_refctd_ptr<const IQueryPool>(queryPool)))
        return false;

    return endQuery_impl(queryPool, query);
}

bool IGPUCommandBuffer::copyQueryPoolResults(video::IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, buffer_t* dstBuffer, size_t dstOffset, size_t stride, core::bitflag<video::IQueryPool::E_QUERY_RESULTS_FLAGS> flags)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!queryPool || !this->isCompatibleDevicewise(queryPool))
        return false;

    if (!dstBuffer || !this->isCompatibleDevicewise(dstBuffer))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyQueryPoolResultsCmd>(m_commandList, core::smart_refctd_ptr<const IQueryPool>(queryPool), core::smart_refctd_ptr<const IGPUBuffer>(dstBuffer)))
        return false;

    return copyQueryPoolResults_impl(queryPool, firstQuery, queryCount, dstBuffer, dstOffset, stride, flags);
}

bool IGPUCommandBuffer::setDeviceMask(uint32_t deviceMask)
{
    if (!checkStateBeforeRecording())
        return false;

    m_deviceMask = deviceMask;
    return setDeviceMask_impl(deviceMask);
}

bool IGPUCommandBuffer::bindGraphicsPipeline(const graphics_pipeline_t* pipeline)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!pipeline || pipeline->getAPIType() != getAPIType())
        return false;

    if (!this->isCompatibleDevicewise(pipeline))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBindGraphicsPipelineCmd>(m_commandList, core::smart_refctd_ptr<const IGPUGraphicsPipeline>(pipeline)))
        return false;

    return bindGraphicsPipeline_impl(pipeline);
}

bool IGPUCommandBuffer::pushConstants(const pipeline_layout_t* layout, core::bitflag<asset::IShader::E_SHADER_STAGE> stageFlags, uint32_t offset, uint32_t size, const void* pValues)
{
    if (!checkStateBeforeRecording())
        return false;

    if (layout->getAPIType() != getAPIType())
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CPushConstantsCmd>(m_commandList, core::smart_refctd_ptr<const IGPUPipelineLayout>(layout)))
        return false;

    pushConstants_impl(layout, stageFlags, offset, size, pValues);

    return true;
}

bool IGPUCommandBuffer::clearColorImage(image_t* image, asset::IImage::LAYOUT imageLayout, const asset::SClearColorValue* pColor, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!image || image->getAPIType() != getAPIType())
        return false;

    if (!this->isCompatibleDevicewise(image))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CClearColorImageCmd>(m_commandList, core::smart_refctd_ptr<const IGPUImage>(image)))
        return false;

    return clearColorImage_impl(image, imageLayout, pColor, rangeCount, pRanges);
}

bool IGPUCommandBuffer::clearDepthStencilImage(image_t* image, asset::IImage::LAYOUT imageLayout, const asset::SClearDepthStencilValue* pDepthStencil, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!image || image->getAPIType() != getAPIType())
        return false;

    if (!this->isCompatibleDevicewise(image))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CClearDepthStencilImageCmd>(m_commandList, core::smart_refctd_ptr<const IGPUImage>(image)))
        return false;

    return clearDepthStencilImage_impl(image, imageLayout, pDepthStencil, rangeCount, pRanges);
}

bool IGPUCommandBuffer::fillBuffer(buffer_t* dstBuffer, size_t dstOffset, size_t size, uint32_t data)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!dstBuffer || dstBuffer->getAPIType() != getAPIType())
        return false;

    if (!this->isCompatibleDevicewise(dstBuffer))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CFillBufferCmd>(m_commandList, core::smart_refctd_ptr<const IGPUBuffer>(dstBuffer)))
        return false;

    return fillBuffer_impl(dstBuffer, dstOffset, size, data);
}

bool IGPUCommandBuffer::bindVertexBuffers(uint32_t firstBinding, uint32_t bindingCount, const buffer_t* const* const pBuffers, const size_t* pOffsets)
{
    if (!checkStateBeforeRecording())
        return false;

    for (uint32_t i = 0u; i < bindingCount; ++i)
    {
        if (pBuffers[i] && !this->isCompatibleDevicewise(pBuffers[i]))
            return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBindVertexBuffersCmd>(m_commandList, firstBinding, bindingCount, pBuffers))
        return false;

    bindVertexBuffers_impl(firstBinding, bindingCount, pBuffers, pOffsets);

    return true;
}

bool IGPUCommandBuffer::dispatchIndirect(const buffer_t* buffer, size_t offset)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!buffer || buffer->getAPIType() != getAPIType())
        return false;

    if (!this->isCompatibleDevicewise(buffer))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CDispatchIndirectCmd>(m_commandList, core::smart_refctd_ptr<const IGPUBuffer>(buffer)))
        return false;

    return dispatchIndirect_impl(buffer, offset);
}

bool IGPUCommandBuffer::setEvent(event_t* _event, const SDependencyInfo& depInfo)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!_event || _event->getAPIType() != getAPIType())
        return false;

    if (!this->isCompatibleDevicewise(_event))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CSetEventCmd>(m_commandList, core::smart_refctd_ptr<const IGPUEvent>(_event)))
        return false;

    return setEvent_impl(_event, depInfo);
}

bool IGPUCommandBuffer::resetEvent(event_t* _event, asset::E_PIPELINE_STAGE_FLAGS stageMask)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!_event || _event->getAPIType() != getAPIType())
        return false;

    if (!this->isCompatibleDevicewise(_event))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CResetEventCmd>(m_commandList, core::smart_refctd_ptr<const IGPUEvent>(_event)))
        return false;

    return resetEvent_impl(_event, stageMask);
}

bool IGPUCommandBuffer::waitEvents(uint32_t eventCount, event_t* const* const pEvents, const SDependencyInfo* depInfo)
{
    if (!checkStateBeforeRecording())
        return false;

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

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CWaitEventsCmd>(m_commandList, depInfo->bufBarrierCount, buffers_raw, depInfo->imgBarrierCount, images_raw, eventCount, pEvents))
        return false;

    return waitEvents_impl(eventCount, pEvents, depInfo);
}

bool IGPUCommandBuffer::drawMeshBuffer(const IGPUMeshBuffer::base_t* meshBuffer)
{
    if (!checkStateBeforeRecording())
        return false;

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
    if (!checkStateBeforeRecording())
        return false;

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

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyBufferCmd>(m_commandList, core::smart_refctd_ptr<const IGPUBuffer>(srcBuffer), core::smart_refctd_ptr<const IGPUBuffer>(dstBuffer)))
        return false;

    return copyBuffer_impl(srcBuffer, dstBuffer, regionCount, pRegions);
}

bool IGPUCommandBuffer::copyImage(const image_t* srcImage, asset::IImage::LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SImageCopy* pRegions)
{
    if (!checkStateBeforeRecording())
        return false;

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

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyImageCmd>(m_commandList, core::smart_refctd_ptr<const IGPUImage>(srcImage), core::smart_refctd_ptr<const IGPUImage>(dstImage)))
        return false;

    return copyImage_impl(srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);
}

bool IGPUCommandBuffer::copyBufferToImage(const buffer_t* srcBuffer, image_t* dstImage, asset::IImage::LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!srcBuffer || srcBuffer->getAPIType() != getAPIType())
        return false;

    if (!dstImage || dstImage->getAPIType() != getAPIType())
        return false;

    if (!this->isCompatibleDevicewise(srcBuffer))
        return false;

    if (!this->isCompatibleDevicewise(dstImage))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyBufferToImageCmd>(m_commandList, core::smart_refctd_ptr<const IGPUBuffer>(srcBuffer), core::smart_refctd_ptr<const IGPUImage>(dstImage)))
        return false;

    return copyBufferToImage_impl(srcBuffer, dstImage, dstImageLayout, regionCount, pRegions);
}

bool IGPUCommandBuffer::blitImage(const image_t* srcImage, asset::IImage::LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageBlit* pRegions, asset::ISampler::E_TEXTURE_FILTER filter)
{
    if (!checkStateBeforeRecording())
        return false;

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
        if (pRegions[i].dstSubresource.aspectMask.value != pRegions[i].srcSubresource.aspectMask.value)
            return false;
        if (pRegions[i].dstSubresource.layerCount != pRegions[i].srcSubresource.layerCount)
            return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBlitImageCmd>(m_commandList, core::smart_refctd_ptr<const IGPUImage>(srcImage), core::smart_refctd_ptr<const IGPUImage>(dstImage)))
        return false;

    return blitImage_impl(srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions, filter);
}

bool IGPUCommandBuffer::copyImageToBuffer(const image_t* srcImage, asset::IImage::LAYOUT srcImageLayout, buffer_t* dstBuffer, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!srcImage || (srcImage->getAPIType() != getAPIType()))
        return false;

    if (!dstBuffer || (dstBuffer->getAPIType() != getAPIType()))
        return false;

    if (!this->isCompatibleDevicewise(srcImage))
        return false;

    if (!this->isCompatibleDevicewise(dstBuffer))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyImageToBufferCmd>(m_commandList, core::smart_refctd_ptr<const IGPUImage>(srcImage), core::smart_refctd_ptr<const IGPUBuffer>(dstBuffer)))
        return false;

    return copyImageToBuffer_impl(srcImage, srcImageLayout, dstBuffer, regionCount, pRegions);
}

bool IGPUCommandBuffer::resolveImage(const image_t* srcImage, asset::IImage::LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageResolve* pRegions)
{
    if (!checkStateBeforeRecording())
        return false;

    if (!srcImage || srcImage->getAPIType() != getAPIType())
        return false;

    if (!dstImage || dstImage->getAPIType() != getAPIType())
        return false;

    if (!this->isCompatibleDevicewise(srcImage))
        return false;

    if (!this->isCompatibleDevicewise(dstImage))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CResolveImageCmd>(m_commandList, core::smart_refctd_ptr<const IGPUImage>(srcImage), core::smart_refctd_ptr<const IGPUImage>(dstImage)))
        return false;

    return resolveImage_impl(srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);
}

bool IGPUCommandBuffer::executeCommands(uint32_t count, cmdbuf_t* const* const cmdbufs)
{
    if (!checkStateBeforeRecording())
        return false;

    for (uint32_t i = 0u; i < count; ++i)
    {
        if (!cmdbufs[i] || (cmdbufs[i]->getLevel() != EL_SECONDARY))
            return false;

        if (!this->isCompatibleDevicewise(cmdbufs[i]))
            return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CExecuteCommandsCmd>(m_commandList, count, cmdbufs))
        return false;

    return executeCommands_impl(count, cmdbufs);
}

}