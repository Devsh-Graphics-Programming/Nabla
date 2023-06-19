#include "nbl/video/IGPUCommandBuffer.h"
#include "nbl/video/IPhysicalDevice.h"

namespace nbl::video
{

bool IGPUCommandBuffer::checkStateBeforeRecording(const core::bitflag<queue_flags_t> allowedQueueFlags, const core::bitflag<RENDERPASS_SCOPE> renderpassScope)
{
    if (m_state!=STATE::RECORDING)
    {
        m_logger.log("Failed to record into command buffer: not in RECORDING state.", system::ILogger::ELL_ERROR);
        return false;
    }
    const bool withinSubpass = m_cachedInheritanceInfo.subpass != SInheritanceInfo{}.subpass;
    if (!renderpassScope.hasFlags(withinSubpass ? RENDERPASS_SCOPE::INSIDE:RENDERPASS_SCOPE::OUTSIDE))
    {
        m_logger.log(
            "Failed to record into command buffer: this command has Renderpass Scope flags %d and you're currently%s recording a Renderpass.", 
            system::ILogger::ELL_ERROR, static_cast<uint32_t>(renderpassScope.value), withinSubpass ? "":" not"
        );
        return false;
    }
    if (checkForParentPoolReset())
    {
        m_logger.log("Failed to record into command buffer: pool was reset since the recording begin() call.", system::ILogger::ELL_ERROR);
        return false;
    }
    const auto& queueFamilyProps = getOriginDevice()->getPhysicalDevice()->getQueueFamilyProperties()[m_cmdpool->getQueueFamilyIndex()];
    if (!bool(queueFamilyProps.queueFlags&allowedQueueFlags))
    {
        m_logger.log("Failed to record into command buffer: this command is not supported by the Queue Family of the Command Pool.", system::ILogger::ELL_ERROR);
        return false;
    }
    return true;
}


bool IGPUCommandBuffer::begin(const core::bitflag<USAGE> flags, const SInheritanceInfo* inheritanceInfo)
{
    // Using Vulkan 1.2 VUIDs here because we don't want to confuse ourselves with Dynamic Rendering being core
    // https://vulkan.lunarg.com/doc/view/1.2.176.1/linux/1.2-extensions/vkspec.html#VUID-vkBeginCommandBuffer-commandBuffer-00049
    if (m_state == STATE::RECORDING || m_state == STATE::PENDING)
    {
        m_logger.log("Failed to begin command buffer: command buffer must not be in RECORDING or PENDING state.", system::ILogger::ELL_ERROR);
        return false;
    }

    const bool whollyInsideRenderpass = flags.hasFlags(USAGE::RENDER_PASS_CONTINUE_BIT);
    const auto physDev = getOriginDevice()->getPhysicalDevice();
    if (m_level==LEVEL::PRIMARY)
    {
        // https://vulkan.lunarg.com/doc/view/1.2.176.1/linux/1.2-extensions/vkspec.html#VUID-vkBeginCommandBuffer-commandBuffer-02840
        if (flags.hasFlags(USAGE::ONE_TIME_SUBMIT_BIT|USAGE::SIMULTANEOUS_USE_BIT))
        {
            m_logger.log("Failed to begin command buffer: a primary command buffer must not have both USAGE::ONE_TIME_SUBMIT_BIT and USAGE::SIMULTANEOUS_USE_BIT set.", system::ILogger::ELL_ERROR);
            return false;
        }
        // this is an extra added by me (devsh)
        if (whollyInsideRenderpass)
        {
            m_logger.log("Failed to begin command buffer: a primary command buffer must not have the USAGE::RENDER_PASS_CONTINUE_BIT set.", system::ILogger::ELL_ERROR);
            return false;
        }
        #ifdef  _NBL_DEBUG
        if (inheritanceInfo)
            m_logger.log("Don't include inheritance info for Primary CommandBuffers!", system::ILogger::ELL_WARNING);
        #endif //  _NBL_DEBUG
    }
    else if (inheritanceInfo)
    {
        // https://vulkan.lunarg.com/doc/view/1.2.176.1/linux/1.2-extensions/vkspec.html#VUID-vkBeginCommandBuffer-commandBuffer-00052
        if (inheritanceInfo->queryFlags.hasFlags(QUERY_CONTROL_FLAGS::PRECISE_BIT) && (!inheritanceInfo->occlusionQueryEnable/*|| TODO: precise occlusion queries limit/feature*/))
        {
            m_logger.log("Failed to begin command buffer: Precise Occlusion Queries cannot be used!", system::ILogger::ELL_ERROR);
            return false;
        }
    }
    // https://vulkan.lunarg.com/doc/view/1.2.176.1/linux/1.2-extensions/vkspec.html#VUID-vkBeginCommandBuffer-commandBuffer-00051
    else
    {
        m_logger.log("Failed to begin command buffer: a secondary command buffer requires an inheritance info structure!", system::ILogger::ELL_ERROR);
        return false;
    }

    if (whollyInsideRenderpass)
    {
        if (!physDev->getQueueFamilyProperties()[m_cmdpool->getQueueFamilyIndex()].queueFlags.hasFlags(queue_flags_t::GRAPHICS_BIT))
        {
            m_logger.log("Failed to begin command buffer: a secondary command buffer which continues a Render Pass is requires a Graphics Queue Family.", system::ILogger::ELL_ERROR);
            return false;
        }

        // https://vulkan.lunarg.com/doc/view/1.2.176.1/linux/1.2-extensions/vkspec.html#VUID-VkCommandBufferBeginInfo-flags-00053
        // https://vulkan.lunarg.com/doc/view/1.2.176.1/linux/1.2-extensions/vkspec.html#VUID-VkCommandBufferBeginInfo-flags-00054
        if (!inheritanceInfo || !inheritanceInfo->renderpass || !inheritanceInfo->renderpass->isCompatibleDevicewise(this) || inheritanceInfo->subpass<inheritanceInfo->renderpass->getSubpasses().size())
        {
            m_logger.log("Failed to begin command buffer: a secondary command buffer must have valid inheritance info with a valid renderpass.", system::ILogger::ELL_ERROR);
            return false;
        }

        // https://vulkan.lunarg.com/doc/view/1.2.176.1/linux/1.2-extensions/vkspec.html#VUID-VkCommandBufferBeginInfo-flags-00055
        if (inheritanceInfo->framebuffer && !inheritanceInfo->framebuffer->isCompatibleDevicewise(this)/* TODO: better check needed || inheritanceInfo->framebuffer->getCreationParameters().renderpass != inheritanceInfo->renderpass*/)
            return false;
    }
    // extras from me (devsh)
    else if (inheritanceInfo->renderpass || inheritanceInfo->framebuffer)
    {
        m_logger.log("Failed to begin command buffer: Do not provide renderpass or framebuffer to a Command Buffer begin without also the USAGE::RENDER_PASS_CONTINUE_BIT bitflag.", system::ILogger::ELL_ERROR);
        return false;
    }

    checkForParentPoolReset();
    m_resetCheckedStamp = m_cmdpool->getResetCounter();

    if (m_state != STATE::INITIAL)
    {
        releaseResourcesBackToPool();
        if (!canReset())
        {
            m_logger.log("Failed to begin command buffer: command buffer allocated from a command pool with ECF_RESET_COMMAND_BUFFER_BIT flag not set cannot be reset, and command buffer not in INITIAL state.", system::ILogger::ELL_ERROR);
            m_state = STATE::INVALID;
            return false;
        }

        m_state = STATE::INITIAL;
    }

    assert(m_state == STATE::INITIAL);

    m_recordingFlags = flags;
    m_state = STATE::RECORDING;
    m_cachedInheritanceInfo = inheritanceInfo ? (*inheritanceInfo):SInheritanceInfo{};
    return begin_impl(flags, inheritanceInfo);
}

bool IGPUCommandBuffer::reset(const core::bitflag<RESET_FLAGS> flags)
{
    if (!canReset())
    {
        m_logger.log("Failed to reset command buffer.", system::ILogger::ELL_ERROR);
        m_state = STATE::INVALID;
        return false;
    }

    if (checkForParentPoolReset())
        return true;

    releaseResourcesBackToPool();
    m_state = STATE::INITIAL;

    return reset_impl(flags);
}

bool IGPUCommandBuffer::end()
{
    const bool whollyInsideRenderpass = m_recordingFlags.hasFlags(USAGE::RENDER_PASS_CONTINUE_BIT);
    if (!checkStateBeforeRecording(whollyInsideRenderpass ? queue_flags_t::GRAPHICS_BIT:queue_flags_t::NONE,whollyInsideRenderpass ? RENDERPASS_SCOPE::INSIDE:RENDERPASS_SCOPE::OUTSIDE))
        return false;

    m_state = STATE::EXECUTABLE;
    return end_impl();
}

/*
bool IGPUCommandBuffer::setDeviceMask(uint32_t deviceMask)
{
    if (!checkStateBeforeRecording())
        return false;

    m_deviceMask = deviceMask;
    return setDeviceMask_impl(deviceMask);
}
*/

bool IGPUCommandBuffer::validateDependency(const SDependencyInfo& depInfo) const
{
    // under NBL_DEBUG, cause waay too expensive to validate
    #ifdef _NBL_DEBUG
    auto device = getOriginDevice();
    for (auto j=0u; j<depInfo.memBarrierCount; j++)
    if (!device->validateMemoryBarrier(m_cmdpool->getQueueFamilyIndex(),depInfos[i].memBarriers[j]))
        return false;
    for (auto j=0u; j<depInfo.bufBarrierCount; j++)
    if (!device->validateMemoryBarrier(m_cmdpool->getQueueFamilyIndex(),depInfos[i].bufBarriers[j]))
        return false;
    for (auto j=0u; j<depInfo.imgBarrierCount; j++)
    if (!device->validateMemoryBarrier(m_cmdpool->getQueueFamilyIndex(),depInfos[i].imgBarriers[j]))
        return false;
    #endif // _NBL_DEBUG
    return true;
}

bool IGPUCommandBuffer::setEvent(IGPUEvent* _event, const SDependencyInfo& depInfo)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (!_event || !this->isCompatibleDevicewise(_event))
        return false;

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdSetEvent2-srcStageMask-03827
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdSetEvent2-srcStageMask-03828
    if (!validateDependency(depInfo))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CSetEventCmd>(m_commandList, core::smart_refctd_ptr<const IGPUEvent>(_event)))
        return false;

    return setEvent_impl(_event,depInfo);
}

bool IGPUCommandBuffer::resetEvent(IGPUEvent* _event, const core::bitflag<stage_flags_t> stageMask)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (!_event || !this->isCompatibleDevicewise(_event))
        return false;

    if (stageMask.hasFlags(stage_flags_t::HOST_BIT))
        return false;

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdResetEvent2-stageMask-03929
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdResetEvent2-stageMask-03930
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdResetEvent2-stageMask-03931
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdResetEvent2-stageMask-03932
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdResetEvent2-stageMask-03934
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdResetEvent2-stageMask-03935
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdResetEvent2-stageMask-07316
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdResetEvent2-stageMask-07346
    if (!getOriginDevice()->supportsMask(m_cmdpool->getQueueFamilyIndex(),stageMask))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CResetEventCmd>(m_commandList,core::smart_refctd_ptr<const IGPUEvent>(_event)))
        return false;

    return resetEvent_impl(_event,stageMask);
}

bool IGPUCommandBuffer::waitEvents(const uint32_t eventCount, IGPUEvent* const* const pEvents, const SDependencyInfo* depInfos)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (eventCount==0u)
        return false;

    uint32_t totalBufferCount = 0u;
    uint32_t totalImageCount = 0u;
    for (auto i=0u; i<eventCount; ++i)
    {
        if (!pEvents[i] || !this->isCompatibleDevicewise(pEvents[i]))
            return false;

        const auto& depInfo = depInfos[i];
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdWaitEvents2-srcStageMask-03842
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdWaitEvents2-srcStageMask-03843
        // TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdWaitEvents2-dependencyFlags-03844
        if (validateDependency(depInfo))
            return false;

        totalBufferCount += depInfo.bufBarrierCount;
        totalImageCount += depInfo.imgBarrierCount;
    }

    auto* cmd = m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CWaitEventsCmd>(m_commandList,eventCount,pEvents,totalBufferCount,totalImageCount);
    if (!cmd)
        return false;

    auto outIt = cmd->getResources();
    for (auto i=0u; i<eventCount; ++i)
    {
        const auto& depInfo = depInfos[i];
        for (auto j=0u; j<depInfo.bufBarrierCount; j++)
            *(outIt++) = depInfo.bufBarriers[j].range.buffer;
        for (auto j=0u; j<depInfo.imgBarrierCount; j++)
            *(outIt++) = depInfo.imgBarriers[j].image;
    }
    return waitEvents_impl(eventCount,pEvents,depInfos);
}

bool IGPUCommandBuffer::pipelineBarrier(const core::bitflag<asset::E_DEPENDENCY_FLAGS> dependencyFlags, const SDependencyInfo& depInfo)
{
    if (!checkStateBeforeRecording(/*everything is allowed*/))
        return false;

    if (depInfo.memBarrierCount==0u && depInfo.bufBarrierCount==0u && depInfo.imgBarrierCount==0u)
        return false;
    
    if (m_cachedInheritanceInfo.subpass!=SInheritanceInfo{}.subpass)
    {
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdPipelineBarrier2-bufferMemoryBarrierCount-01178
        if (depInfo.bufBarrierCount)
            return false;
        for (auto i=0u; i<depInfo.imgBarrierCount; i++)
        {
            const auto& barrier = depInfo.imgBarriers[i];
            // Cannot do barriers on anything thats not an attachment, and only subpass deps can transition layouts!
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdPipelineBarrier2-oldLayout-01181
            if (barrier.newLayout!=barrier.oldLayout)
                return false;

            // TODO: under NBL_DEBUG, cause waay too expensive to validate
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdPipelineBarrier2-image-04073
        }
        // TODO: under NBL_DEBUG, cause waay too expensive to validate
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdPipelineBarrier2-None-07889
    }

    auto* cmd = m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CPipelineBarrierCmd>(m_commandList,depInfo.bufBarrierCount,depInfo.imgBarrierCount);
    if (!cmd)
        return false;

    auto outIt = cmd->getResources();
    for (auto j=0u; j<depInfo.bufBarrierCount; j++)
        *(outIt++) = depInfo.bufBarriers[j].range.buffer;
    for (auto j=0u; j<depInfo.imgBarrierCount; j++)
        *(outIt++) = depInfo.imgBarriers[j].image;
    return pipelineBarrier_impl(dependencyFlags,depInfo);
}


bool IGPUCommandBuffer::fillBuffer(IGPUBuffer* dstBuffer, size_t dstOffset, size_t size, uint32_t data)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT|queue_flags_t::TRANSFER_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;
    
    if (!validate_updateBuffer<true>(dstBuffer,dstOffset,size))
    {
        m_logger.log("Invalid arguments see `IGPUCommandBuffer::validate_updateBuffer`.", system::ILogger::ELL_ERROR);
        return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CFillBufferCmd>(m_commandList,core::smart_refctd_ptr<const IGPUBuffer>(dstBuffer)))
        return false;

    return fillBuffer_impl(dstBuffer, dstOffset, size, data);
}

bool IGPUCommandBuffer::updateBuffer(IGPUBuffer* const dstBuffer, const size_t dstOffset, const size_t dataSize, const void* const pData)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT|queue_flags_t::TRANSFER_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (!validate_updateBuffer(dstBuffer,dstOffset,dataSize))
    {
        m_logger.log("Invalid arguments see `IGPUCommandBuffer::validate_updateBuffer`.", system::ILogger::ELL_ERROR);
        return false;
    }
    if (dataSize>0x10000ull)
    {
        m_logger.log("Inline Buffer Updates are limited to 64kb!", system::ILogger::ELL_ERROR);
        return false;
    }
    if (!dstBuffer->getCreationParams().usage.hasFlags(IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF))
    {
        m_logger.log("You need the `IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF` usage flag which is missing!", system::ILogger::ELL_ERROR);
        return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CUpdateBufferCmd>(m_commandList,core::smart_refctd_ptr<const IGPUBuffer>(dstBuffer)))
        return false;

    return updateBuffer_impl(dstBuffer,dstOffset,dataSize,pData);
}

bool IGPUCommandBuffer::copyBuffer(const IGPUBuffer* const srcBuffer, IGPUBuffer* const dstBuffer, uint32_t regionCount, const SBufferCopy* const pRegions)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT|queue_flags_t::TRANSFER_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdCopyBuffer.html#VUID-vkCmdCopyBuffer-srcBuffer-parameter
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdCopyBuffer.html#VUID-vkCmdCopyBuffer-srcBuffer-00118
    if (!srcBuffer || !this->isCompatibleDevicewise(srcBuffer) || !srcBuffer->getCreationParams().usage.hasFlags(IGPUBuffer::EUF_TRANSFER_DST_BIT))
        return false;
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdCopyBuffer.html#VUID-vkCmdCopyBuffer-dstBuffer-parameter
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdCopyBuffer.html#VUID-vkCmdCopyBuffer-dstBuffer-00120
    if (!dstBuffer || !this->isCompatibleDevicewise(dstBuffer) || !dstBuffer->getCreationParams().usage.hasFlags(IGPUBuffer::EUF_TRANSFER_DST_BIT))
        return false;
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdCopyBuffer.html#VUID-vkCmdCopyBuffer-regionCount-arraylength
    if (regionCount==0u)
        return false;

    // pRegions is too expensive to validate

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyBufferCmd>(m_commandList,core::smart_refctd_ptr<const IGPUBuffer>(srcBuffer),core::smart_refctd_ptr<const IGPUBuffer>(dstBuffer)))
        return false;

    return copyBuffer_impl(srcBuffer, dstBuffer, regionCount, pRegions);
}


bool IGPUCommandBuffer::clearColorImage(IGPUImage* const image, const IGPUImage::LAYOUT imageLayout, const SClearColorValue* const pColor, const uint32_t rangeCount, const IGPUImage::SSubresourceRange* const pRanges)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (!image || !this->isCompatibleDevicewise(image))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CClearColorImageCmd>(m_commandList,core::smart_refctd_ptr<const IGPUImage>(image)))
        return false;
    return clearColorImage_impl(image, imageLayout, pColor, rangeCount, pRanges);
}

bool IGPUCommandBuffer::clearDepthStencilImage(IGPUImage* const image, const IGPUImage::LAYOUT imageLayout, const SClearDepthStencilValue* const pDepthStencil, const uint32_t rangeCount, const IGPUImage::SSubresourceRange* const pRanges)
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (!image || !this->isCompatibleDevicewise(image))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CClearDepthStencilImageCmd>(m_commandList,core::smart_refctd_ptr<const IGPUImage>(image)))
        return false;
    return clearDepthStencilImage_impl(image, imageLayout, pDepthStencil, rangeCount, pRanges);
}

bool IGPUCommandBuffer::copyBufferToImage(const IGPUBuffer* const srcBuffer, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const IGPUImage::SBufferCopy* const pRegions)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT|queue_flags_t::TRANSFER_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (!srcBuffer || !this->isCompatibleDevicewise(srcBuffer))
        return false;
    if (!dstImage || !this->isCompatibleDevicewise(dstImage))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyBufferToImageCmd>(m_commandList, core::smart_refctd_ptr<const IGPUBuffer>(srcBuffer), core::smart_refctd_ptr<const IGPUImage>(dstImage)))
        return false;

    return copyBufferToImage_impl(srcBuffer, dstImage, dstImageLayout, regionCount, pRegions);
}

bool IGPUCommandBuffer::copyImageToBuffer(const IGPUBuffer* const srcImage, const IGPUImage::LAYOUT srcImageLayout, const IGPUBuffer* const dstBuffer, const uint32_t regionCount, const IGPUImage::SBufferCopy* const pRegions)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT|queue_flags_t::TRANSFER_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (!srcImage || !this->isCompatibleDevicewise(srcImage))
        return false;
    if (!dstBuffer || !this->isCompatibleDevicewise(dstBuffer))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyImageToBufferCmd>(m_commandList, core::smart_refctd_ptr<const IGPUImage>(srcImage), core::smart_refctd_ptr<const IGPUBuffer>(dstBuffer)))
        return false;

    return copyImageToBuffer_impl(srcImage, srcImageLayout, dstBuffer, regionCount, pRegions);
}

bool IGPUCommandBuffer::copyImage(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const IGPUImage::SImageCopy* const pRegions)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT|queue_flags_t::TRANSFER_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (!srcImage || !this->isCompatibleDevicewise(srcImage))
        return false;
    if (!dstImage || !this->isCompatibleDevicewise(dstImage))
        return false;

    if (!dstImage->validateCopies(pRegions, pRegions + regionCount, srcImage))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyImageCmd>(m_commandList, core::smart_refctd_ptr<const IGPUImage>(srcImage), core::smart_refctd_ptr<const IGPUImage>(dstImage)))
        return false;

    return copyImage_impl(srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);
}


bool IGPUCommandBuffer::copyAccelerationStructure(const IGPUAccelerationStructure::CopyInfo& copyInfo)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::TRANSFER_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (!copyInfo.src || !this->isCompatibleDevicewise(copyInfo.src))
        return false;
    if (!copyInfo.dst || !this->isCompatibleDevicewise(copyInfo.dst))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyAccelerationStructureCmd>(m_commandList, core::smart_refctd_ptr<const IGPUAccelerationStructure>(copyInfo.src), core::smart_refctd_ptr<const IGPUAccelerationStructure>(copyInfo.dst)))
        return false;

    return copyAccelerationStructure_impl(copyInfo);
}

bool IGPUCommandBuffer::copyAccelerationStructureToMemory(const IGPUAccelerationStructure::DeviceCopyToMemoryInfo& copyInfo)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::TRANSFER_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (!copyInfo.src || !this->isCompatibleDevicewise(copyInfo.src))
        return false;
    if (!copyInfo.dst.buffer || !this->isCompatibleDevicewise(copyInfo.dst.buffer.get()))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyAccelerationStructureToOrFromMemoryCmd>(m_commandList, core::smart_refctd_ptr<const IGPUAccelerationStructure>(copyInfo.src), core::smart_refctd_ptr<const IGPUBuffer>(copyInfo.dst.buffer)))
        return false;

    return copyAccelerationStructureToMemory_impl(copyInfo);
}

bool IGPUCommandBuffer::copyAccelerationStructureFromMemory(const IGPUAccelerationStructure::DeviceCopyFromMemoryInfo& copyInfo)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::TRANSFER_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;
    
    if (!copyInfo.src.buffer || !this->isCompatibleDevicewise(copyInfo.src.buffer.get()))
        return false;
    if (!copyInfo.dst || !this->isCompatibleDevicewise(copyInfo.dst))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyAccelerationStructureToOrFromMemoryCmd>(m_commandList, core::smart_refctd_ptr<const IGPUAccelerationStructure>(copyInfo.dst), core::smart_refctd_ptr<const IGPUBuffer>(copyInfo.src.buffer)))
        return false;

    return copyAccelerationStructureFromMemory_impl(copyInfo);
}

#if 0
bool IGPUCommandBuffer::buildAccelerationStructures(const core::SRange<const IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, const IGPUAccelerationStructure::BuildRangeInfo* const* const ppBuildRangeInfos)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT,RENDERPASS_SCOPE::OUTSIDE))
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

bool IGPUCommandBuffer::buildAccelerationStructuresIndirect(const core::SRange<const IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, const core::SRange<const IGPUAccelerationStructure::DeviceAddressType>& pIndirectDeviceAddresses, const uint32_t* const pIndirectStrides, const uint32_t* const* const ppMaxPrimitiveCounts)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT,RENDERPASS_SCOPE::OUTSIDE))
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
#endif

bool IGPUCommandBuffer::bindComputePipeline(const IGPUComputePipeline* const pipeline)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT))
        return false;

    if (!this->isCompatibleDevicewise(pipeline))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBindComputePipelineCmd>(m_commandList, core::smart_refctd_ptr<const IGPUComputePipeline>(pipeline)))
        return false;

    bindComputePipeline_impl(pipeline);

    return true;
}

bool IGPUCommandBuffer::bindGraphicsPipeline(const IGPUGraphicsPipeline* const pipeline)
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT))
        return false;

    if (!pipeline || !this->isCompatibleDevicewise(pipeline))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBindGraphicsPipelineCmd>(m_commandList, core::smart_refctd_ptr<const IGPUGraphicsPipeline>(pipeline)))
        return false;

    return bindGraphicsPipeline_impl(pipeline);
}

bool IGPUCommandBuffer::bindDescriptorSets(
    const asset::E_PIPELINE_BIND_POINT pipelineBindPoint, const IGPUPipelineLayout* const layout,
    const uint32_t firstSet, const uint32_t descriptorSetCount, const IGPUDescriptorSet* const* const pDescriptorSets,
    const uint32_t dynamicOffsetCount = 0u, const uint32_t* const dynamicOffsets)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT))
        return false;

    if (!layout ||!this->isCompatibleDevicewise(layout))
        return false;

    for (uint32_t i=0u; i<descriptorSetCount; ++i)
    if (pDescriptorSets[i])
    {
        if (!this->isCompatibleDevicewise(pDescriptorSets[i]))
        {
            m_logger.log("IGPUCommandBuffer::bindDescriptorSets failed, pDescriptorSets[%d] was not created by the same ILogicalDevice as the commandbuffer!", system::ILogger::ELL_ERROR, i);
            return false;
        }
        if (!pDescriptorSets[i]->getLayout()->isIdenticallyDefined(layout->getDescriptorSetLayout(firstSet+i)))
        {
            m_logger.log("IGPUCommandBuffer::bindDescriptorSets failed, pDescriptorSets[%d] not identically defined as layout's %dth descriptor layout!", system::ILogger::ELL_ERROR, i, firstSet+i);
            return false;
        }
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBindDescriptorSetsCmd>(m_commandList,core::smart_refctd_ptr<const IGPUPipelineLayout>(layout),descriptorSetCount,pDescriptorSets))
        return false;

    for (uint32_t i=0u; i<descriptorSetCount; ++i)
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

                m_state = STATE::INVALID;
                return false;
            }
        }
        else
            m_boundDescriptorSetsRecord.insert({ pDescriptorSets[i], currentVersion });
    }

    return bindDescriptorSets_impl(pipelineBindPoint, layout, firstSet, descriptorSetCount, pDescriptorSets, dynamicOffsetCount, dynamicOffsets);
}

bool IGPUCommandBuffer::pushConstants(const IGPUPipelineLayout* const layout, const core::bitflag<IGPUShader::E_SHADER_STAGE> stageFlags, const uint32_t offset, const uint32_t size, const void* const pValues)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT))
        return false;

    if (!layout || this->isCompatibleDevicewise(layout))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CPushConstantsCmd>(m_commandList, core::smart_refctd_ptr<const IGPUPipelineLayout>(layout)))
        return false;

    return pushConstants_impl(layout, stageFlags, offset, size, pValues);
}

bool IGPUCommandBuffer::bindVertexBuffers(const uint32_t firstBinding, const uint32_t bindingCount, const IGPUBuffer* const* const pBuffers, const size_t* const pOffsets)
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT))
        return false;

    for (uint32_t i = 0u; i < bindingCount; ++i)
    if (pBuffers[i] && !this->isCompatibleDevicewise(pBuffers[i]))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBindVertexBuffersCmd>(m_commandList, firstBinding, bindingCount, pBuffers))
        return false;

    return bindVertexBuffers_impl(firstBinding, bindingCount, pBuffers, pOffsets);
}

bool IGPUCommandBuffer::bindIndexBuffer(const IGPUBuffer* const buffer, const size_t offset, const asset::E_INDEX_TYPE indexType)
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT))
        return false;

    if (!buffer || !this->isCompatibleDevicewise(buffer))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBindIndexBufferCmd>(m_commandList, core::smart_refctd_ptr<const IGPUBuffer>(buffer)))
        return false;

    return bindIndexBuffer_impl(buffer, offset, indexType);
}


bool IGPUCommandBuffer::resetQueryPool(IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount)
{
    // also encode/decode and opticalflow (video) queues
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (!queryPool || !this->isCompatibleDevicewise(queryPool))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CResetQueryPoolCmd>(m_commandList, core::smart_refctd_ptr<const IQueryPool>(queryPool)))
        return false;

    return resetQueryPool_impl(queryPool, firstQuery, queryCount);
}

bool IGPUCommandBuffer::beginQuery(IQueryPool* const queryPool, const uint32_t query, const core::bitflag<QUERY_CONTROL_FLAGS> flags)
{
    // also encode/decode and opticalflow (video) queues
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT))
        return false;

    if (!queryPool || !this->isCompatibleDevicewise(queryPool))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBeginQueryCmd>(m_commandList, core::smart_refctd_ptr<const IQueryPool>(queryPool)))
        return false;

    return beginQuery_impl(queryPool, query, flags);
}

bool IGPUCommandBuffer::endQuery(IQueryPool* const queryPool, const uint32_t query)
{
    // also encode/decode and opticalflow (video) queues
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT))
        return false;

    if (!queryPool || !this->isCompatibleDevicewise(queryPool))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CEndQueryCmd>(m_commandList, core::smart_refctd_ptr<const IQueryPool>(queryPool)))
        return false;

    return endQuery_impl(queryPool, query);
}

bool IGPUCommandBuffer::writeTimestamp(const stage_flags_t pipelineStage, IQueryPool* const queryPool, const uint32_t query)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT|queue_flags_t::TRANSFER_BIT))
        return false;

    const auto qFamIx = m_cmdpool->getQueueFamilyIndex();
    if (!getOriginDevice()->getSupportedStageMask(qFamIx).hasFlags(pipelineStage) || getOriginDevice()->getPhysicalDevice()->getQueueFamilyProperties()[qFamIx].timestampValidBits==0u)
        return false;

    if (!queryPool || !this->isCompatibleDevicewise(queryPool) || queryPool->getCreationParameters().queryType!=IQueryPool::EQT_TIMESTAMP || query>=queryPool->getCreationParameters().queryCount)
        return false;

    assert(core::isPoT(static_cast<uint32_t>(pipelineStage))); // should only be 1 stage (1 bit set)

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CWriteTimestampCmd>(m_commandList, core::smart_refctd_ptr<const IQueryPool>(queryPool)))
        return false;

    return writeTimestamp_impl(pipelineStage, queryPool, query);
}

#if 0
bool IGPUCommandBuffer::writeAccelerationStructureProperties(const core::SRange<const IGPUAccelerationStructure>& pAccelerationStructures, const IQueryPool::E_QUERY_TYPE queryType, IQueryPool* const queryPool, const uint32_t firstQuery)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (!queryPool || !this->isCompatibleDevicewise(queryPool) || pAccelerationStructures.empty())
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
#endif

bool IGPUCommandBuffer::copyQueryPoolResults(
    const IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount,
    IGPUBuffer* const dstBuffer, const size_t dstOffset, const size_t stride, const core::bitflag<IQueryPool::E_QUERY_RESULTS_FLAGS> flags)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    // TODO: rest of validation
    if (!queryPool || !this->isCompatibleDevicewise(queryPool) || firstQuery+queryCount>=queryPool->getCreationParameters().queryCount)
        return false;

    if (flags.hasFlags(IQueryPool::E_QUERY_RESULTS_FLAGS::EQRF_64_BIT))
    {
        if (dstOffset&0x7u)
            return false;
    }
    else if (dstOffset&0x3u)
        return false;

    const auto& params = dstBuffer->getCreationParams();
    if (!dstBuffer || !this->isCompatibleDevicewise(dstBuffer) || params.usage.hasFlags(IGPUBuffer::EUF_TRANSFER_DST_BIT) || dstOffset+queryCount*stride>=params.size)
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyQueryPoolResultsCmd>(m_commandList, core::smart_refctd_ptr<const IQueryPool>(queryPool), core::smart_refctd_ptr<const IGPUBuffer>(dstBuffer)))
        return false;

    return copyQueryPoolResults_impl(queryPool, firstQuery, queryCount, dstBuffer, dstOffset, stride, flags);
}
#if 0
bool IGPUCommandBuffer::drawIndirect(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride)
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::INSIDE))
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
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::INSIDE))
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
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::INSIDE))
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
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::INSIDE))
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
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    const auto apiType = getAPIType();
    if ((apiType != pRenderPassBegin->renderpass->getAPIType()) || (apiType != pRenderPassBegin->framebuffer->getAPIType()))
        return false;

    if (!this->isCompatibleDevicewise(pRenderPassBegin->framebuffer.get()))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBeginRenderPassCmd>(m_commandList, core::smart_refctd_ptr<const IGPURenderpass>(pRenderPassBegin->renderpass), core::smart_refctd_ptr<const IGPUFramebuffer>(pRenderPassBegin->framebuffer)))
        return false;

    m_cachedInheritanceInfo.subpass = 0;
    return beginRenderPass_impl(pRenderPassBegin, content);
}

bool IGPUCommandBuffer::endRenderPass()
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::INSIDE))
        return false;

    m_cachedInheritanceInfo.subpass = SInheritanceInfo{}.subpass;
    return endRenderPass_impl();
}

static void getResourcesFromBuildGeometryInfos(const core::SRange<IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, core::vector<core::smart_refctd_ptr<const IGPUAccelerationStructure>>& accelerationStructures, core::vector<core::smart_refctd_ptr<const IGPUBuffer>>& buffers)
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

    const IGPUBuffer* gpuBufferBindings[asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT];
    {
        for (size_t i = 0; i < asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; ++i)
            gpuBufferBindings[i] = vertexBufferBindings[i].buffer.get();
    }

    size_t bufferBindingsOffsets[asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT];
    {
        for (size_t i = 0; i < asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; ++i)
            bufferBindingsOffsets[i] = vertexBufferBindings[i].offset;
    }

    bindVertexBuffers(0, asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT, gpuBufferBindings, bufferBindingsOffsets);
    bindIndexBuffer(indexBufferBinding.buffer.get(), indexBufferBinding.offset, indexType);

    const bool isIndexed = indexType != asset::EIT_UNKNOWN;

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
#endif

static bool disallowedLayoutForBlitAndResolve(const IGPUImage::LAYOUT layout)
{
    switch (layout)
    {
        case IGPUImage::LAYOUT::GENERAL:
        case IGPUImage::LAYOUT::TRANSFER_SRC_OPTIMAL:
        case IGPUImage::LAYOUT::SHARED_PRESENT:
            break;
        default:
            return false;
    }
    return true;
}

bool IGPUCommandBuffer::blitImage(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const SImageBlit* pRegions, const IGPUSampler::E_TEXTURE_FILTER filter)
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (regionCount==0u || !pRegions || disallowedLayoutForBlitAndResolve(srcImageLayout) || disallowedLayoutForBlitAndResolve(dstImageLayout))
        return false;

    const auto* physDev = getOriginDevice()->getPhysicalDevice();
    const auto& srcParams = srcImage->getCreationParameters();
    if (!srcImage || !this->isCompatibleDevicewise(srcImage) || !srcParams.usage.hasFlags(IGPUImage::EUF_TRANSFER_SRC_BIT) || !physDev->getImageFormatUsages(srcImage->getTiling())[srcParams.format].blitSrc)
        return false;

    const auto& dstParams = dstImage->getCreationParameters();
    if (!dstImage || !this->isCompatibleDevicewise(dstImage) || !dstParams.usage.hasFlags(IGPUImage::EUF_TRANSFER_SRC_BIT) || !physDev->getImageFormatUsages(dstImage->getTiling())[dstParams.format].blitDst)
        return false;

    // TODO rest of: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdBlitImage.html#VUID-vkCmdBlitImage-srcImage-00229

    for (uint32_t i=0u; i<regionCount; ++i)
    {
        if (pRegions[i].dstSubresource.aspectMask!=pRegions[i].srcSubresource.aspectMask)
            return false;
        if (pRegions[i].dstSubresource.layerCount!=pRegions[i].srcSubresource.layerCount)
            return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBlitImageCmd>(m_commandList, core::smart_refctd_ptr<const IGPUImage>(srcImage), core::smart_refctd_ptr<const IGPUImage>(dstImage)))
        return false;

    return blitImage_impl(srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions, filter);
}

bool IGPUCommandBuffer::resolveImage(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const SImageResolve* pRegions)
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;
    
    if (regionCount==0u || !pRegions || disallowedLayoutForBlitAndResolve(srcImageLayout) || disallowedLayoutForBlitAndResolve(dstImageLayout))
        return false;

    const auto* physDev = getOriginDevice()->getPhysicalDevice();
    const auto& srcParams = srcImage->getCreationParameters();
    if (!srcImage || !this->isCompatibleDevicewise(srcImage) || !srcParams.usage.hasFlags(IGPUImage::EUF_TRANSFER_SRC_BIT))
        return false;
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdResolveImage.html#VUID-vkCmdResolveImage-srcImage-00258
    if (srcParams.samples==IGPUImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT)
        return false;

    const auto& dstParams = dstImage->getCreationParameters();
    if (!dstImage || !this->isCompatibleDevicewise(dstImage) || !dstParams.usage.hasFlags(IGPUImage::EUF_TRANSFER_SRC_BIT) || !physDev->getImageFormatUsages(dstImage->getTiling())[dstParams.format].attachment)
        return false;
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdResolveImage.html#VUID-vkCmdResolveImage-dstImage-00259
    if (dstParams.samples!=IGPUImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT)
        return false;

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdResolveImage.html#VUID-vkCmdResolveImage-srcImage-01386
    if (srcParams.format!=dstParams.format)
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CResolveImageCmd>(m_commandList, core::smart_refctd_ptr<const IGPUImage>(srcImage), core::smart_refctd_ptr<const IGPUImage>(dstImage)))
        return false;

    return resolveImage_impl(srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);
}

bool IGPUCommandBuffer::executeCommands(const uint32_t count, IGPUCommandBuffer* const* const cmdbufs)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT|queue_flags_t::TRANSFER_BIT))
        return false;

    for (uint32_t i=0u; i<count; ++i)
    {
        if (!cmdbufs[i] || cmdbufs[i]->getLevel()!=LEVEL::SECONDARY)
            return false;

        if (!this->isCompatibleDevicewise(cmdbufs[i]))
            return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CExecuteCommandsCmd>(m_commandList,count,cmdbufs))
        return false;

    return executeCommands_impl(count,cmdbufs);
}

}