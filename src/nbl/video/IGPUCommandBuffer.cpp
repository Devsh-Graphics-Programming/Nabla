#define _NBL_VIDEO_I_GPU_COMMAND_BUFFER_CPP_
#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/IPhysicalDevice.h"

namespace nbl::video
{
    
IGPUCommandBuffer::IGPUCommandBuffer(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const LEVEL lvl, core::smart_refctd_ptr<IGPUCommandPool>&& _cmdpool, system::logger_opt_smart_ptr&& logger)
    : IBackendObject(std::move(dev)), m_cmdpool(_cmdpool), m_logger(std::move(logger)), m_level(lvl)
{
}

bool IGPUCommandBuffer::checkStateBeforeRecording(const core::bitflag<queue_flags_t> allowedQueueFlags, const core::bitflag<RENDERPASS_SCOPE> renderpassScope)
{
    if (m_state!=STATE::RECORDING)
    {
        m_logger.log("Failed to record into command buffer: not in RECORDING state.", system::ILogger::ELL_ERROR);
        return false;
    }
    const bool withinSubpass = m_cachedInheritanceInfo.subpass!=SInheritanceInfo{}.subpass;
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
        if (!inheritanceInfo || !inheritanceInfo->renderpass || !inheritanceInfo->renderpass->isCompatibleDevicewise(this) || inheritanceInfo->subpass<inheritanceInfo->renderpass->getSubpassCount())
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

bool IGPUCommandBuffer::invalidDependency(const SDependencyInfo& depInfo) const
{
    // under NBL_DEBUG, cause waay too expensive to validate
    #ifdef _NBL_DEBUG
    auto device = getOriginDevice();
    for (auto j=0u; j<depInfo.memBarrierCount; j++)
    if (!device->validateMemoryBarrier(m_cmdpool->getQueueFamilyIndex(),depInfo.memBarriers[j]))
        return true;
    for (auto j=0u; j<depInfo.bufBarrierCount; j++)
    {
        if (invalidBufferRange(depInfo.bufBarriers[j].range))
        if (!device->validateMemoryBarrier(m_cmdpool->getQueueFamilyIndex(),depInfo.bufBarriers[j]))
            return true;
    }
    for (auto j=0u; j<depInfo.imgBarrierCount; j++)
    if (!device->validateMemoryBarrier(m_cmdpool->getQueueFamilyIndex(),depInfo.imgBarriers[j]))
        return true;
    #endif // _NBL_DEBUG
    return false;
}

bool IGPUCommandBuffer::setEvent(IGPUEvent* _event, const SDependencyInfo& depInfo)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (!_event || !this->isCompatibleDevicewise(_event))
        return false;

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdSetEvent2-srcStageMask-03827
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdSetEvent2-srcStageMask-03828
    if (invalidDependency(depInfo))
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
        if (invalidDependency(depInfo))
            return false;

        totalBufferCount += depInfo.bufBarrierCount;
        totalImageCount += depInfo.imgBarrierCount;
    }

    auto* cmd = m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CWaitEventsCmd>(m_commandList,eventCount,pEvents,totalBufferCount,totalImageCount);
    if (!cmd)
        return false;

    auto outIt = cmd->getDeviceMemoryBacked();
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

    if (invalidDependency(depInfo))
        return false;
    
    const bool withinSubpass = m_cachedInheritanceInfo.subpass!=SInheritanceInfo{}.subpass;
    if (withinSubpass)
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

    auto outIt = cmd->getVariableCountResources();
    for (auto j=0u; j<depInfo.bufBarrierCount; j++)
        *(outIt++) = depInfo.bufBarriers[j].range.buffer;
    for (auto j=0u; j<depInfo.imgBarrierCount; j++)
        *(outIt++) = depInfo.imgBarriers[j].image;
    return pipelineBarrier_impl(dependencyFlags,depInfo);
}


bool IGPUCommandBuffer::fillBuffer(const asset::SBufferRange<IGPUBuffer>& range, uint32_t data)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT|queue_flags_t::TRANSFER_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;
    
    if (invalidBufferRange(range,4u,IGPUBuffer::EUF_TRANSFER_DST_BIT))
    {
        m_logger.log("Invalid arguments see `IGPUCommandBuffer::invalidBufferRange`.", system::ILogger::ELL_ERROR);
        return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CFillBufferCmd>(m_commandList,core::smart_refctd_ptr<const IGPUBuffer>(range.buffer)))
        return false;
    return fillBuffer_impl(range,data);
}

bool IGPUCommandBuffer::updateBuffer(const asset::SBufferRange<IGPUBuffer>& range, const void* const pData)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT|queue_flags_t::TRANSFER_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;
    
    if (invalidBufferRange(range,4u,IGPUBuffer::EUF_TRANSFER_DST_BIT|IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF))
    {
        m_logger.log("Invalid arguments see `IGPUCommandBuffer::validate_updateBuffer`.", system::ILogger::ELL_ERROR);
        return false;
    }
    if (range.actualSize()>0x10000ull)
    {
        m_logger.log("Inline Buffer Updates are limited to 64kb!", system::ILogger::ELL_ERROR);
        return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CUpdateBufferCmd>(m_commandList,core::smart_refctd_ptr<const IGPUBuffer>(range.buffer)))
        return false;
    return updateBuffer_impl(range,pData);
}

bool IGPUCommandBuffer::copyBuffer(const IGPUBuffer* const srcBuffer, IGPUBuffer* const dstBuffer, uint32_t regionCount, const SBufferCopy* const pRegions)
{
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdCopyBuffer.html#VUID-vkCmdCopyBuffer-regionCount-arraylength
    if (regionCount==0u)
        return false;

    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT|queue_flags_t::TRANSFER_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdCopyBuffer.html#VUID-vkCmdCopyBuffer-srcBuffer-parameter
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdCopyBuffer.html#VUID-vkCmdCopyBuffer-srcBuffer-00118
    if (invalidBuffer(srcBuffer,IGPUBuffer::EUF_TRANSFER_SRC_BIT))
        return false;
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdCopyBuffer.html#VUID-vkCmdCopyBuffer-dstBuffer-parameter
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdCopyBuffer.html#VUID-vkCmdCopyBuffer-dstBuffer-00120
    if (invalidBuffer(dstBuffer,IGPUBuffer::EUF_TRANSFER_DST_BIT))
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

    if (invalidDestinationImage<true>(image,imageLayout))
        return false;
    const auto format = image->getCreationParameters().format;
    if (asset::isDepthOrStencilFormat(format) || asset::isBlockCompressionFormat(format))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CClearColorImageCmd>(m_commandList,core::smart_refctd_ptr<const IGPUImage>(image)))
        return false;
    return clearColorImage_impl(image, imageLayout, pColor, rangeCount, pRanges);
}

bool IGPUCommandBuffer::clearDepthStencilImage(IGPUImage* const image, const IGPUImage::LAYOUT imageLayout, const SClearDepthStencilValue* const pDepthStencil, const uint32_t rangeCount, const IGPUImage::SSubresourceRange* const pRanges)
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;
    
    if (invalidDestinationImage<true>(image,imageLayout))
        return false;
    const auto format = image->getCreationParameters().format;
    if (!asset::isDepthOrStencilFormat(format))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CClearDepthStencilImageCmd>(m_commandList,core::smart_refctd_ptr<const IGPUImage>(image)))
        return false;
    return clearDepthStencilImage_impl(image, imageLayout, pDepthStencil, rangeCount, pRanges);
}

bool IGPUCommandBuffer::copyBufferToImage(const IGPUBuffer* const srcBuffer, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const IGPUImage::SBufferCopy* const pRegions)
{
    if (regionCount==0u)
        return false;
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT|queue_flags_t::TRANSFER_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;
    
    if (invalidBuffer(srcBuffer,IGPUBuffer::EUF_TRANSFER_SRC_BIT))
        return false;
    if (invalidDestinationImage(dstImage,dstImageLayout))
        return false;

    // pRegions is too expensive to validate

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyBufferToImageCmd>(m_commandList, core::smart_refctd_ptr<const IGPUBuffer>(srcBuffer), core::smart_refctd_ptr<const IGPUImage>(dstImage)))
        return false;

    return copyBufferToImage_impl(srcBuffer, dstImage, dstImageLayout, regionCount, pRegions);
}

bool IGPUCommandBuffer::copyImageToBuffer(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, const IGPUBuffer* const dstBuffer, const uint32_t regionCount, const IGPUImage::SBufferCopy* const pRegions)
{
    if (regionCount==0u)
        return false;
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT|queue_flags_t::TRANSFER_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;
    
    if (invalidSourceImage(srcImage,srcImageLayout))
        return false;
    if (invalidBuffer(dstBuffer,IGPUBuffer::EUF_TRANSFER_DST_BIT))
        return false;

    // pRegions is too expensive to validate

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyImageToBufferCmd>(m_commandList, core::smart_refctd_ptr<const IGPUImage>(srcImage), core::smart_refctd_ptr<const IGPUBuffer>(dstBuffer)))
        return false;

    return copyImageToBuffer_impl(srcImage, srcImageLayout, dstBuffer, regionCount, pRegions);
}

bool IGPUCommandBuffer::copyImage(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const IGPUImage::SImageCopy* const pRegions)
{
    if (regionCount==0u)
        return false;
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT|queue_flags_t::TRANSFER_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (invalidSourceImage(srcImage,srcImageLayout))
        return false;
    if (invalidDestinationImage(dstImage,dstImageLayout))
        return false;

    const auto& srcParams = srcImage->getCreationParameters();
    const auto& dstParams = dstImage->getCreationParameters();
    if (srcParams.samples!=dstParams.samples)
        return false;
    if (asset::getBytesPerPixel(srcParams.format)!=asset::getBytesPerPixel(dstParams.format))
        return false;

    // pRegions is too expensive to validate

    if (!dstImage->validateCopies(pRegions,pRegions+regionCount,srcImage))
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
    if (invalidBufferBinding(copyInfo.dst,256u,IGPUBuffer::EUF_TRANSFER_DST_BIT))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyAccelerationStructureToOrFromMemoryCmd>(m_commandList, core::smart_refctd_ptr<const IGPUAccelerationStructure>(copyInfo.src), core::smart_refctd_ptr<const IGPUBuffer>(copyInfo.dst.buffer)))
        return false;

    return copyAccelerationStructureToMemory_impl(copyInfo);
}

bool IGPUCommandBuffer::copyAccelerationStructureFromMemory(const IGPUAccelerationStructure::DeviceCopyFromMemoryInfo& copyInfo)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::TRANSFER_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;
    
    if (invalidBufferBinding(copyInfo.src,256u,IGPUBuffer::EUF_TRANSFER_SRC_BIT))
        return false;
    if (!copyInfo.dst || !this->isCompatibleDevicewise(copyInfo.dst))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyAccelerationStructureToOrFromMemoryCmd>(m_commandList, core::smart_refctd_ptr<const IGPUAccelerationStructure>(copyInfo.dst), core::smart_refctd_ptr<const IGPUBuffer>(copyInfo.src.buffer)))
        return false;

    return copyAccelerationStructureFromMemory_impl(copyInfo);
}

bool IGPUCommandBuffer::buildAccelerationStructures(const core::SRange<const IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, const IGPUAccelerationStructure::BuildRangeInfo* const* const ppBuildRangeInfos)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;
    
    const uint32_t resourceCount = validateBuildGeometryInfos(pInfos);
    if (resourceCount==0u)
        return false;

    auto cmd = m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBuildAccelerationStructuresCmd>(m_commandList,resourceCount);
    if (!cmd)
        return false;

    cmd->fill(pInfos);
    return buildAccelerationStructures_impl(pInfos, ppBuildRangeInfos);
}

bool IGPUCommandBuffer::buildAccelerationStructuresIndirect(const core::SRange<const IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, const core::SRange<const IGPUAccelerationStructure::DeviceAddressType>& pIndirectDeviceAddresses, const uint32_t* const pIndirectStrides, const uint32_t* const* const ppMaxPrimitiveCounts)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;
    
    const uint32_t resourceCount = validateBuildGeometryInfos(pInfos);
    if (resourceCount==0u)
        return false;

    auto cmd = m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBuildAccelerationStructuresCmd>(m_commandList,resourceCount);
    if (!cmd)
        return false;

    cmd->fill(pInfos);
    return buildAccelerationStructuresIndirect_impl(pInfos, pIndirectDeviceAddresses, pIndirectStrides, ppMaxPrimitiveCounts);
}


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

bool IGPUCommandBuffer::bindVertexBuffers(const uint32_t firstBinding, const uint32_t bindingCount, const asset::SBufferBinding<const IGPUBuffer>* const pBindings)
{
    if (firstBinding+bindingCount>asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT)
        return false;
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT))
        return false;

    for (uint32_t i=0u; i<bindingCount; ++i)
    if (pBindings[i].buffer && invalidBufferBinding(pBindings[i],4u/*or should we derive from component format?*/,IGPUBuffer::EUF_VERTEX_BUFFER_BIT))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBindVertexBuffersCmd>(m_commandList,pBindings))
        return false;

    return bindVertexBuffers_impl(firstBinding, bindingCount, pBindings);
}

bool IGPUCommandBuffer::bindIndexBuffer(const asset::SBufferBinding<const IGPUBuffer>& binding, const asset::E_INDEX_TYPE indexType)
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT))
        return false;

    if (binding.buffer)
    {
        size_t alignment;
        switch (indexType)
        {
            case asset::EIT_16BIT:
                alignment = alignof(uint16_t);
                break;
            case asset::EIT_32BIT:
                alignment = alignof(uint32_t);
                break;
            default:
                return false;
        }
        if (invalidBufferBinding(binding,alignment,IGPUBuffer::EUF_INDEX_BUFFER_BIT))
            return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBindIndexBufferCmd>(m_commandList, core::smart_refctd_ptr<const IGPUBuffer>(binding.buffer)))
        return false;

    return bindIndexBuffer_impl(binding,indexType);
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

bool IGPUCommandBuffer::writeAccelerationStructureProperties(const core::SRange<const IGPUAccelerationStructure*>& pAccelerationStructures, const IQueryPool::E_QUERY_TYPE queryType, IQueryPool* const queryPool, const uint32_t firstQuery)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (!queryPool || !this->isCompatibleDevicewise(queryPool) || pAccelerationStructures.empty())
        return false;

    for (auto& as : pAccelerationStructures)
    if (invalidAccelerationStructure(as))
        return false;

    auto cmd = m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CWriteAccelerationStructurePropertiesCmd>(m_commandList, queryPool, pAccelerationStructures.size());
    if (!cmd)
        return false;

    auto oit = cmd->getVariableCountResources();
    for (auto& as : pAccelerationStructures)
        *(oit++) = core::smart_refctd_ptr<const core::IReferenceCounted>(as);
    return writeAccelerationStructureProperties_impl(pAccelerationStructures, queryType, queryPool, firstQuery);
}

bool IGPUCommandBuffer::copyQueryPoolResults(
    const IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount,
    const asset::SBufferBinding<const IGPUBuffer>& dstBuffer, const size_t stride, const core::bitflag<IQueryPool::E_QUERY_RESULTS_FLAGS> flags)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    // TODO: rest of validation
    if (!queryPool || !this->isCompatibleDevicewise(queryPool) || queryCount==0u || firstQuery+queryCount>=queryPool->getCreationParameters().queryCount)
        return false;

    const size_t alignment = flags.hasFlags(IQueryPool::E_QUERY_RESULTS_FLAGS::EQRF_64_BIT) ? alignof(uint64_t):alignof(uint32_t);
    if (invalidBufferRange({dstBuffer.offset,queryCount*stride,dstBuffer.buffer},alignment,IGPUBuffer::EUF_TRANSFER_DST_BIT))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyQueryPoolResultsCmd>(m_commandList, core::smart_refctd_ptr<const IQueryPool>(queryPool), core::smart_refctd_ptr<const IGPUBuffer>(dstBuffer.buffer)))
        return false;

    return copyQueryPoolResults_impl(queryPool, firstQuery, queryCount, dstBuffer, stride, flags);
}


bool IGPUCommandBuffer::dispatch(const uint32_t groupCountX, const uint32_t groupCountY, const uint32_t groupCountZ)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (groupCountX==0 || groupCountY==0 || groupCountZ==0)
        return false;

    const auto& limits = getOriginDevice()->getPhysicalDevice()->getLimits();
    if (groupCountX>limits.maxComputeWorkGroupCount[0] || groupCountY>limits.maxComputeWorkGroupCount[1] || groupCountZ>limits.maxComputeWorkGroupCount[2])
        return false;

    return dispatch_impl(groupCountX,groupCountY,groupCountZ);
}

bool IGPUCommandBuffer::dispatchIndirect(const asset::SBufferBinding<const IGPUBuffer>& binding)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (invalidBufferBinding(binding,4u/*TODO: is it really 4?*/,IGPUBuffer::EUF_INDIRECT_BUFFER_BIT))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CIndirectCmd>(m_commandList,core::smart_refctd_ptr<const IGPUBuffer>(binding.buffer)))
        return false;

    return dispatchIndirect_impl(binding);
}


bool IGPUCommandBuffer::beginRenderPass(const SRenderpassBeginInfo& info, const SUBPASS_CONTENTS contents)
{
    if (m_recordingFlags.hasFlags(USAGE::RENDER_PASS_CONTINUE_BIT))
        return false;
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (!info.framebuffer || !this->isCompatibleDevicewise(info.framebuffer))
        return false;

    const auto& renderArea = info.renderArea;
    if (renderArea.extent.width==0u || renderArea.extent.height==0u)
        return false;

    const auto& params = info.framebuffer->getCreationParameters();
    if (renderArea.offset.x+renderArea.extent.width>params.width || renderArea.offset.y+renderArea.extent.height>params.height)
        return false;

    const auto rp = params.renderpass;
    if (rp->getDepthStencilLoadOpAttachmentEnd()!=0u && !info.depthStencilClearValues)
        return false;
    if (rp->getColorLoadOpAttachmentEnd()!=0u && !info.colorClearValues)
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBeginRenderPassCmd>(m_commandList,core::smart_refctd_ptr<const IGPUFramebuffer>(info.framebuffer)))
        return false;

    m_cachedInheritanceInfo.renderpass = params.renderpass;
    m_cachedInheritanceInfo.subpass = 0;
    m_cachedInheritanceInfo.framebuffer = core::smart_refctd_ptr<const IGPUFramebuffer>(info.framebuffer);
    return beginRenderPass_impl(info,contents);
}

bool IGPUCommandBuffer::nextSubpass(const SUBPASS_CONTENTS contents)
{
    if (m_recordingFlags.hasFlags(USAGE::RENDER_PASS_CONTINUE_BIT))
        return false;
    if (m_cachedInheritanceInfo.subpass>=m_cachedInheritanceInfo.renderpass->getSubpassCount())
        return false;

    m_cachedInheritanceInfo.subpass++;
    return nextSubpass_impl(contents);
}

bool IGPUCommandBuffer::endRenderPass()
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::INSIDE))
        return false;

    m_cachedInheritanceInfo.subpass = SInheritanceInfo{}.subpass;
    return endRenderPass_impl();
}


bool IGPUCommandBuffer::clearAttachments(const SClearAttachments& info)
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::INSIDE))
        return false;

    auto invalidRegion = [this](const SClearAttachments::SRegion& region)->bool
    {
        if (region.used())
        {
            if (region.rect.extent.width==0u || region.rect.extent.height==0u || region.layerCount==0u)
                return true;
        
            // cannot validate without tracking more stuff
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdClearAttachments.html#VUID-vkCmdClearAttachments-pRects-00016
            // TODO: if (m_cachedInheritanceInfo.renderpass->getUsesMultiview()) then, instead check https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdClearAttachments.html#VUID-vkCmdClearAttachments-baseArrayLayer-00018
            if (region.baseArrayLayer+region.layerCount>m_cachedInheritanceInfo.framebuffer->getCreationParameters().layers)
                return true;
        }
        return true;
    };

    if (bool(info.depthStencilAspectMask)!=info.depthStencilRegion.used() || invalidRegion(info.depthStencilRegion))
        return false;
    for (auto i=0u; i<IGPURenderpass::SCreationParams::SSubpassDescription::MaxColorAttachments; i++)
    if (invalidRegion(info.colorRegions[i]))
        return false;

    return clearAttachments_impl(info);
}


bool IGPUCommandBuffer::draw(const uint32_t vertexCount, const uint32_t instanceCount, const uint32_t firstVertex, const uint32_t firstInstance)
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::INSIDE))
        return false;

    if (vertexCount==0u || instanceCount == 0u)
        return false;

    return draw_impl(vertexCount,instanceCount,firstVertex,firstInstance);
}

bool IGPUCommandBuffer::drawIndexed(const uint32_t indexCount, const uint32_t instanceCount, const uint32_t firstIndex, const int32_t vertexOffset, const uint32_t firstInstance)
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::INSIDE))
        return false;

    if (indexCount==0u || instanceCount == 0u)
        return false;

    return drawIndexed_impl(indexCount,instanceCount,firstIndex,vertexOffset,firstInstance);
}

template<typename IndirectCommand>
bool IGPUCommandBuffer::invalidDrawIndirect(const asset::SBufferBinding<const IGPUBuffer>& binding, const uint32_t drawCount, uint32_t stride)
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::INSIDE))
        return true;

    if (drawCount)
    {
        if (drawCount==1u)
            stride = sizeof(IndirectCommand);
        if (stride&0x3u || stride<sizeof(IndirectCommand))
            return true;
        if (drawCount>getOriginDevice()->getPhysicalDevice()->getLimits().maxDrawIndirectCount)
            return true;
        if (invalidBufferRange({binding.offset,stride*(drawCount-1u)+sizeof(IndirectCommand),binding.buffer},alignof(uint32_t),IGPUBuffer::EUF_INDIRECT_BUFFER_BIT))
            return true;
    }
    return false;
}

template<typename IndirectCommand>
bool IGPUCommandBuffer::invalidDrawIndirectCount(const asset::SBufferBinding<const IGPUBuffer>& indirectBinding, const asset::SBufferBinding<const IGPUBuffer>& countBinding, const uint32_t maxDrawCount, const uint32_t stride)
{
    if (!getOriginDevice()->getPhysicalDevice()->getLimits().drawIndirectCount)
        return true;

    if (invalidDrawIndirect<IndirectCommand>(indirectBinding,countBinding,maxDrawCount,stride))
        return true;
    if (invalidBufferRange({countBinding.offset,sizeof(uint32_t),countBinding.buffer},alignof(uint32_t),IGPUBuffer::EUF_INDIRECT_BUFFER_BIT))
        return true;

    return false;
}

bool IGPUCommandBuffer::drawIndirect(const asset::SBufferBinding<const IGPUBuffer>& binding, const uint32_t drawCount, const uint32_t stride)
{
    if (invalidDrawIndirect<asset::DrawArraysIndirectCommand_t>(binding,drawCount,stride))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CIndirectCmd>(m_commandList,core::smart_refctd_ptr<const IGPUBuffer>(binding.buffer)))
        return false;

    return drawIndirect_impl(binding, drawCount, stride);
}

bool IGPUCommandBuffer::drawIndexedIndirect(const asset::SBufferBinding<const IGPUBuffer>& binding, const uint32_t drawCount, const uint32_t stride)
{
    if (invalidDrawIndirect<asset::DrawElementsIndirectCommand_t>(binding,drawCount,stride))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CIndirectCmd>(m_commandList, core::smart_refctd_ptr<const IGPUBuffer>(binding.buffer)))
        return false;

    return drawIndexedIndirect_impl(binding, drawCount, stride);
}

bool IGPUCommandBuffer::drawIndirectCount(const asset::SBufferBinding<const IGPUBuffer>& indirectBinding, const asset::SBufferBinding<const IGPUBuffer>& countBinding, const uint32_t maxDrawCount, const uint32_t stride)
{
    if (!invalidDrawIndirectCount<asset::DrawArraysIndirectCommand_t>(indirectBinding,countBinding,maxDrawCount,stride))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CDrawIndirectCountCmd>(m_commandList, core::smart_refctd_ptr<const IGPUBuffer>(indirectBinding.buffer), core::smart_refctd_ptr<const IGPUBuffer>(countBinding.buffer)))
        return false;

    return drawIndirectCount_impl(indirectBinding, countBinding, maxDrawCount, stride);
}

bool IGPUCommandBuffer::drawIndexedIndirectCount(const asset::SBufferBinding<const IGPUBuffer>& indirectBinding, const asset::SBufferBinding<const IGPUBuffer>& countBinding, const uint32_t maxDrawCount, const uint32_t stride)
{
    if (!invalidDrawIndirectCount<asset::DrawElementsIndirectCommand_t>(indirectBinding,countBinding,maxDrawCount,stride))
        return false;
    
    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CDrawIndirectCountCmd>(m_commandList, core::smart_refctd_ptr<const IGPUBuffer>(indirectBinding.buffer), core::smart_refctd_ptr<const IGPUBuffer>(countBinding.buffer)))
        return false;

    return drawIndexedIndirectCount_impl(indirectBinding, countBinding, maxDrawCount, stride);
}


bool IGPUCommandBuffer::drawMeshBuffer(const IGPUMeshBuffer* const meshBuffer)
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::INSIDE))
        return false;

    if (meshBuffer && meshBuffer->getInstanceCount()==0u)
        return false;

    const auto indexType = meshBuffer->getIndexType();

    if (!bindVertexBuffers(0, asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT, meshBuffer->getVertexBufferBindings()))
        return false;
    if (!bindIndexBuffer(meshBuffer->getIndexBufferBinding(), indexType))
        return false;

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

    auto cmd = m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CExecuteCommandsCmd>(m_commandList,count);
    if (!cmd)
        return false;
    for (auto i=0u; i<count; i++)
        cmd->getVariableCountResources()[i] = core::smart_refctd_ptr<const core::IReferenceCounted>(cmdbufs[i]);
    return executeCommands_impl(count,cmdbufs);
}

}