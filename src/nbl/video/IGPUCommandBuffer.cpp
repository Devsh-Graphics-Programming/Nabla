#define _NBL_VIDEO_I_GPU_COMMAND_BUFFER_CPP_
#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/IPhysicalDevice.h"

#define NBL_LOG_FUNCTION m_logger.log
#include "nbl/logging_macros.h"

namespace nbl::video
{
    
IGPUCommandBuffer::IGPUCommandBuffer(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const IGPUCommandPool::BUFFER_LEVEL lvl, core::smart_refctd_ptr<IGPUCommandPool>&& _cmdpool, system::logger_opt_smart_ptr&& logger)
    : IBackendObject(std::move(dev)), m_cmdpool(_cmdpool), m_logger(std::move(logger)), m_level(lvl)
{
    // prevent false positives on first `begin()`
    m_resetCheckedStamp = m_cmdpool->getResetCounter();
}

bool IGPUCommandBuffer::checkStateBeforeRecording(const core::bitflag<queue_flags_t> allowedQueueFlags, const core::bitflag<RENDERPASS_SCOPE> renderpassScope)
{
    if (m_state!=STATE::RECORDING)
    {
        NBL_LOG_ERROR("not in RECORDING state!");
        return false;
    }
    const bool withinSubpass = m_cachedInheritanceInfo.subpass!=SInheritanceInfo{}.subpass;
    if (!renderpassScope.hasFlags(withinSubpass ? RENDERPASS_SCOPE::INSIDE:RENDERPASS_SCOPE::OUTSIDE))
    {
        NBL_LOG_ERROR(
            "this command has Renderpass Scope flags %d and you're currently%s recording a Renderpass!", 
            system::ILogger::ELL_ERROR, static_cast<uint32_t>(renderpassScope.value), withinSubpass ? "":" not"
        );
        return false;
    }
    if (checkForParentPoolReset())
    {
        NBL_LOG_ERROR("pool was reset since the recording begin() call!");
        return false;
    }
    const auto& queueFamilyProps = getOriginDevice()->getPhysicalDevice()->getQueueFamilyProperties()[m_cmdpool->getQueueFamilyIndex()];
    if (!bool(queueFamilyProps.queueFlags&allowedQueueFlags))
    {
        NBL_LOG_ERROR("this command is not supported by the Queue Family of the Command Pool!");
        return false;
    }
    return true;
}


bool IGPUCommandBuffer::begin(const core::bitflag<USAGE> flags, const SInheritanceInfo* inheritanceInfo)
{
    // Using Vulkan 1.2 VUIDs here because we don't want to confuse ourselves with Dynamic Rendering being core
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBeginCommandBuffer-commandBuffer-00049
    if (m_state == STATE::RECORDING || m_state == STATE::PENDING)
    {
        NBL_LOG_ERROR("command buffer must not be in RECORDING or PENDING state!");
        return false;
    }

    const bool whollyInsideRenderpass = flags.hasFlags(USAGE::RENDER_PASS_CONTINUE_BIT);
    const auto physDev = getOriginDevice()->getPhysicalDevice();
    if (m_level==IGPUCommandPool::BUFFER_LEVEL::PRIMARY)
    {
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBeginCommandBuffer-commandBuffer-02840
        if (flags.hasFlags(USAGE::ONE_TIME_SUBMIT_BIT|USAGE::SIMULTANEOUS_USE_BIT))
        {
            NBL_LOG_ERROR("a primary command buffer must not have both USAGE::ONE_TIME_SUBMIT_BIT and USAGE::SIMULTANEOUS_USE_BIT set!");
            return false;
        }
        // this is an extra added by me (devsh)
        if (whollyInsideRenderpass)
        {
            NBL_LOG_ERROR("a primary command buffer must not have the USAGE::RENDER_PASS_CONTINUE_BIT set!");
            return false;
        }
        #ifdef  _NBL_DEBUG
        if (inheritanceInfo)
            m_logger.log("Don't include inheritance info for Primary CommandBuffers!", system::ILogger::ELL_WARNING);
        #endif //  _NBL_DEBUG
    }
    else if (inheritanceInfo)
    {
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBeginCommandBuffer-commandBuffer-00052
        if (inheritanceInfo->queryFlags.hasFlags(QUERY_CONTROL_FLAGS::PRECISE_BIT) && (!inheritanceInfo->occlusionQueryEnable/*|| TODO: precise occlusion queries limit/feature*/))
        {
            NBL_LOG_ERROR("Precise Occlusion Queries cannot be used!");
            return false;
        }
    }
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBeginCommandBuffer-commandBuffer-00051
    else
    {
        NBL_LOG_ERROR("a secondary command buffer requires an inheritance info structure!");
        return false;
    }

    if (whollyInsideRenderpass)
    {
        if (!physDev->getQueueFamilyProperties()[m_cmdpool->getQueueFamilyIndex()].queueFlags.hasFlags(queue_flags_t::GRAPHICS_BIT))
        {
            NBL_LOG_ERROR("a secondary command buffer which continues a Render Pass is requires a Graphics Queue Family!");
            return false;
        }

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkCommandBufferBeginInfo-flags-00053
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkCommandBufferBeginInfo-flags-00054
        if (!inheritanceInfo || !inheritanceInfo->renderpass || !inheritanceInfo->renderpass->isCompatibleDevicewise(this) || inheritanceInfo->subpass<inheritanceInfo->renderpass->getSubpassCount())
        {
            NBL_LOG_ERROR("a secondary command buffer must have valid inheritance info with a valid renderpass!");
            return false;
        }

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkCommandBufferBeginInfo-flags-00055
        if (inheritanceInfo->framebuffer && !inheritanceInfo->framebuffer->isCompatibleDevicewise(this)/* TODO: better check needed || inheritanceInfo->framebuffer->getCreationParameters().renderpass != inheritanceInfo->renderpass*/)
        {
            NBL_LOG_ERROR("a secondary command buffer must have compatible framebuffer!");
            return false;
        }
    }
    // extras from me (devsh)
    else if (inheritanceInfo && (inheritanceInfo->renderpass||inheritanceInfo->framebuffer))
    {
        NBL_LOG_ERROR("Do not provide renderpass or framebuffer to a Command Buffer begin without also the USAGE::RENDER_PASS_CONTINUE_BIT bitflag!");
        return false;
    }

    checkForParentPoolReset();

    // still not initial and pool wasn't reset
    if (m_state!=STATE::INITIAL)
    {
        releaseResourcesBackToPool();
        if (!canReset())
        {
            NBL_LOG_ERROR("command buffer allocated from a command pool with ECF_RESET_COMMAND_BUFFER_BIT flag not set cannot be reset, and command buffer not in INITIAL state!");
            m_state = STATE::INVALID;
            return false;
        }

        m_state = STATE::INITIAL;
    }

    // still not initial (we're trying to single-reset a commandbuffer that cannot be individually reset)
    // should have been caught out above
    assert(m_state == STATE::INITIAL);

    m_recordingFlags = flags;
    m_state = STATE::RECORDING;
    if (inheritanceInfo)
    {
        if (inheritanceInfo->framebuffer && !inheritanceInfo->framebuffer->getCreationParameters().renderpass->compatible(inheritanceInfo->renderpass))
        {
            NBL_LOG_ERROR("a secondary command buffer must have compatible renderpass!");
            return false;
        }
        if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBeginRenderPassCmd>(m_commandList,core::smart_refctd_ptr<const IGPURenderpass>(inheritanceInfo->renderpass),core::smart_refctd_ptr<const IGPUFramebuffer>(inheritanceInfo->framebuffer)))
        {
            NBL_LOG_ERROR("out of host memory!");
            return false;
        }
        m_cachedInheritanceInfo = *inheritanceInfo;
    }
    else
        m_cachedInheritanceInfo = {};
    m_noCommands = true;
    return begin_impl(flags,inheritanceInfo);
}

bool IGPUCommandBuffer::reset(const core::bitflag<RESET_FLAGS> flags)
{
    if (!canReset())
    {
        NBL_LOG_ERROR("%d!", (uint32_t)m_state);
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
    auto allowedQueueCaps = queue_flags_t::GRAPHICS_BIT;
    if (!whollyInsideRenderpass)
        allowedQueueCaps |= queue_flags_t::COMPUTE_BIT|queue_flags_t::TRANSFER_BIT;
    if (!checkStateBeforeRecording(allowedQueueCaps,whollyInsideRenderpass ? RENDERPASS_SCOPE::INSIDE:RENDERPASS_SCOPE::OUTSIDE))
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

template<typename ResourceBarrier>
bool IGPUCommandBuffer::invalidDependency(const SDependencyInfo<ResourceBarrier>& depInfo) const
{
    // under NBL_DEBUG, cause waay too expensive to validate
    #ifdef _NBL_DEBUG
    auto device = getOriginDevice();
    // TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdPipelineBarrier2-None-07890
    // TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdPipelineBarrier2-None-07891
    // TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdPipelineBarrier2-None-07892
    // TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkBufferMemoryBarrier2-srcStageMask-03851
    for (const auto& barrier : depInfo.memBarriers)
    if (!device->validateMemoryBarrier(m_cmdpool->getQueueFamilyIndex(),barrier))
        return true;
    for (const auto& barrier : depInfo.bufBarriers)
    {
        // AFAIK, no special constraints on alignment or usage here
        if (invalidBufferRange(barrier.range,1u,IGPUBuffer::EUF_NONE))
        if (!device->validateMemoryBarrier(m_cmdpool->getQueueFamilyIndex(),barrier))
            return true;
    }
    for (const auto& barrier : depInfo.imgBarriers)
    if (!device->validateMemoryBarrier(m_cmdpool->getQueueFamilyIndex(),barrier))
        return true;
    #endif // _NBL_DEBUG
    return false;
}
template bool IGPUCommandBuffer::invalidDependency(const SDependencyInfo<asset::SMemoryBarrier>&) const;
template bool IGPUCommandBuffer::invalidDependency(const SDependencyInfo<IGPUCommandBuffer::SOwnershipTransferBarrier>&) const;

bool IGPUCommandBuffer::setEvent(IEvent* _event, const SEventDependencyInfo& depInfo)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (!_event || !this->isCompatibleDevicewise(_event))
    {
        NBL_LOG_ERROR("incompatible event!");
        return false;
    }

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdSetEvent2-srcStageMask-03827
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdSetEvent2-srcStageMask-03828
    if (invalidDependency(depInfo))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CSetEventCmd>(m_commandList, core::smart_refctd_ptr<const IEvent>(_event)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_noCommands = false;
    return setEvent_impl(_event,depInfo);
}

bool IGPUCommandBuffer::resetEvent(IEvent* _event, const core::bitflag<stage_flags_t> stageMask)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (!_event || !this->isCompatibleDevicewise(_event))
    {
        NBL_LOG_ERROR("incompatible event!");
        return false;
    }

    if (stageMask.hasFlags(stage_flags_t::HOST_BIT))
    {
        NBL_LOG_ERROR("stageMask must not include HOST_BIT!");
        return false;
    }

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdResetEvent2-stageMask-03929
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdResetEvent2-stageMask-03930
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdResetEvent2-stageMask-03931
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdResetEvent2-stageMask-03932
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdResetEvent2-stageMask-03934
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdResetEvent2-stageMask-03935
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdResetEvent2-stageMask-07316
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdResetEvent2-stageMask-07346
    if (!getOriginDevice()->supportsMask(m_cmdpool->getQueueFamilyIndex(),stageMask))
    {
        NBL_LOG_ERROR("unsupported stageMask!");
        return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CResetEventCmd>(m_commandList,core::smart_refctd_ptr<const IEvent>(_event)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_noCommands = false;
    return resetEvent_impl(_event,stageMask);
}

bool IGPUCommandBuffer::waitEvents(const std::span<IEvent*> events, const SEventDependencyInfo* depInfos)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::BOTH))
        return false;

    if (events.empty())
    {
        NBL_LOG_ERROR("no events to wait for!");
        return false;
    }

    uint32_t totalBufferCount = 0u;
    uint32_t totalImageCount = 0u;
    for (auto i=0u; i<events.size(); ++i)
    {
        if (!events[i] || !this->isCompatibleDevicewise(events[i]))
        {
            NBL_LOG_ERROR("incompatible event!");
            return false;
        }

        const auto& depInfo = depInfos[i];
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdWaitEvents2-srcStageMask-03842
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdWaitEvents2-srcStageMask-03843
        // TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdWaitEvents2-dependencyFlags-03844
        if (invalidDependency(depInfo))
            return false;

        totalBufferCount += depInfo.bufBarriers.size();
        totalImageCount += depInfo.imgBarriers.size();
    }

    auto* cmd = m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CWaitEventsCmd>(m_commandList,events.size(),events.data(),totalBufferCount,totalImageCount);
    if (!cmd)
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    auto outIt = cmd->getDeviceMemoryBacked();
    for (auto i=0u; i<events.size(); ++i)
    {
        const auto& depInfo = depInfos[i];
        for (const auto& barrier : depInfo.bufBarriers)
            *(outIt++) = barrier.range.buffer;
        for (const auto& barrier : depInfo.imgBarriers)
            *(outIt++) = core::smart_refctd_ptr<const IGPUImage>(barrier.image);
    }
    m_noCommands = false;
    return waitEvents_impl(events,depInfos);
}

bool IGPUCommandBuffer::pipelineBarrier(const core::bitflag<asset::E_DEPENDENCY_FLAGS> dependencyFlags, const SPipelineBarrierDependencyInfo& depInfo)
{
    if (!checkStateBeforeRecording(/*everything is allowed*/))
        return false;

    if (depInfo.memBarriers.empty() && depInfo.bufBarriers.empty() && depInfo.imgBarriers.empty())
    {
        NBL_LOG_ERROR("no dependency info is provided!");
        return false;
    }

    if (invalidDependency(depInfo))
        return false;
    
    const bool withinSubpass = m_cachedInheritanceInfo.subpass!=SInheritanceInfo{}.subpass;
    if (withinSubpass)
    {
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdPipelineBarrier2-bufferMemoryBarrierCount-01178
        if (!depInfo.bufBarriers.empty())
        {
            NBL_LOG_ERROR("buffer memory barriers must be empty while within a subpass!");
            return false;
        }

        auto invalidSubpassMemoryBarrier = [dependencyFlags](nbl::system::logger_opt_smart_ptr logger, const asset::SMemoryBarrier& barrier) -> bool
        {
            if (barrier.srcStageMask&stage_flags_t::FRAMEBUFFER_SPACE_BITS)
            {
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdPipelineBarrier2-None-07890
                if (barrier.dstStageMask&(~stage_flags_t::FRAMEBUFFER_SPACE_BITS))
                {
                    logger.log("destination stage masks of memory barriers included non FRAMEBUFFER_SPACE_BITS stages while source stage masks has FRAMEBUFFER_SPACE_BITS within a subpass!");
                    return true;
                }
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdPipelineBarrier2-dependencyFlags-07891
                if (!dependencyFlags.hasFlags(asset::EDF_BY_REGION_BIT))
                {
                    logger.log("dependency flags of memory barriers must includ EDF_BY_REGION_BIT while source stage masks has FRAMEBUFFER_SPACE_BITS within a subpass!");
                    return true;
                }
            }
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdPipelineBarrier2-None-07892
            constexpr auto NotGraphicsBits = ~stage_flags_t::ALL_GRAPHICS_BITS;
            if ((barrier.srcStageMask&NotGraphicsBits) || (barrier.dstStageMask&NotGraphicsBits)) {
                logger.log("source & destination stage masks of memory barriers must only include graphics pipeline stages!");
                return true;
            }
            return false;
        };
        for (const auto& barrier : depInfo.memBarriers)
        {
            if (invalidSubpassMemoryBarrier(m_logger, barrier))
                return false;
        }
        for (const auto& barrier : depInfo.imgBarriers)
        {
            if (invalidSubpassMemoryBarrier(m_logger, barrier.barrier.dep))
                return false;

            // TODO: under NBL_DEBUG, cause waay too expensive to validate
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdPipelineBarrier2-image-04073

            // Cannot do barriers on anything thats not an attachment, and only subpass deps can transition layouts!
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdPipelineBarrier2-oldLayout-01181
            if (barrier.newLayout!=barrier.oldLayout)
            {
                NBL_LOG_ERROR("can't transit layouts while within a subpass!");
                return false;
            }

            // Ownership Transfers CANNOT HAPPEN MID-RENDERPASS
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdPipelineBarrier2-srcQueueFamilyIndex-01182
            if (barrier.barrier.otherQueueFamilyIndex!=IQueue::FamilyIgnored)
            {
                NBL_LOG_ERROR("can't transfer queue family ownership mid-renderpass!");
                return false;
            }
        }
        // TODO: under NBL_DEBUG, cause waay too expensive to validate
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdPipelineBarrier2-None-07889
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdPipelineBarrier2-None-07893
    }
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdPipelineBarrier2-dependencyFlags-01186
    else if (dependencyFlags.hasFlags(asset::EDF_VIEW_LOCAL_BIT))
    {
        NBL_LOG_ERROR("the dependency flags must not include EDF_VIEW_LOCAL_BIT while not within a subpass!");
        return false;
    }

    auto* cmd = m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CPipelineBarrierCmd>(m_commandList,depInfo.bufBarriers.size(),depInfo.imgBarriers.size());
    if (!cmd)
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    auto outIt = cmd->getVariableCountResources();
    for (const auto& barrier : depInfo.bufBarriers)
        *(outIt++) = barrier.range.buffer;
    for (const auto& barrier : depInfo.imgBarriers)
        *(outIt++) = core::smart_refctd_ptr<const IGPUImage>(barrier.image);
    m_noCommands = false;
    return pipelineBarrier_impl(dependencyFlags,depInfo);
}


bool IGPUCommandBuffer::fillBuffer(const asset::SBufferRange<IGPUBuffer>& range, uint32_t data)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT|queue_flags_t::TRANSFER_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;
    
    if (invalidBufferRange(range,4u,IGPUBuffer::EUF_TRANSFER_DST_BIT))
    {
        NBL_LOG_ERROR("Invalid arguments see `IGPUCommandBuffer::invalidBufferRange`!");
        return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CFillBufferCmd>(m_commandList,core::smart_refctd_ptr<const IGPUBuffer>(range.buffer)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }
    m_noCommands = false;
    return fillBuffer_impl(range,data);
}

bool IGPUCommandBuffer::updateBuffer(const asset::SBufferRange<IGPUBuffer>& range, const void* const pData)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT|queue_flags_t::TRANSFER_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;
    
    if (invalidBufferRange(range,4u,IGPUBuffer::EUF_TRANSFER_DST_BIT|IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF))
    {
        NBL_LOG_ERROR("Invalid arguments see `IGPUCommandBuffer::validate_updateBuffer`!");
        return false;
    }
    if (range.actualSize()>0x10000ull)
    {
        NBL_LOG_ERROR("Inline Buffer Updates are limited to 64kb!");
        return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CUpdateBufferCmd>(m_commandList,core::smart_refctd_ptr<const IGPUBuffer>(range.buffer)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }
    m_noCommands = false;
    return updateBuffer_impl(range,pData);
}

bool IGPUCommandBuffer::copyBuffer(const IGPUBuffer* const srcBuffer, IGPUBuffer* const dstBuffer, uint32_t regionCount, const SBufferCopy* const pRegions)
{
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdCopyBuffer.html#VUID-vkCmdCopyBuffer-regionCount-arraylength
    if (regionCount==0u)
    {
        NBL_LOG_ERROR("regionCount must be larger than 0!");
        return false;
    }

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
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }
    m_noCommands = false;
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
    {
        NBL_LOG_ERROR("invalid format!");
        return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CClearColorImageCmd>(m_commandList,core::smart_refctd_ptr<const IGPUImage>(image)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }
    m_noCommands = false;
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
    {
        NBL_LOG_ERROR("invalid format!");
        return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CClearDepthStencilImageCmd>(m_commandList,core::smart_refctd_ptr<const IGPUImage>(image)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }
    m_noCommands = false;
    return clearDepthStencilImage_impl(image, imageLayout, pDepthStencil, rangeCount, pRanges);
}

bool IGPUCommandBuffer::copyBufferToImage(const IGPUBuffer* const srcBuffer, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const IGPUImage::SBufferCopy* const pRegions)
{
    if (regionCount==0u)
    {
        NBL_LOG_ERROR("regionCount must be larger than 0!");
        return false;
    }

    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT|queue_flags_t::TRANSFER_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;
    
    if (invalidBuffer(srcBuffer,IGPUBuffer::EUF_TRANSFER_SRC_BIT))
        return false;
    if (invalidDestinationImage(dstImage,dstImageLayout))
        return false;

    // pRegions is too expensive to validate

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyBufferToImageCmd>(m_commandList, core::smart_refctd_ptr<const IGPUBuffer>(srcBuffer), core::smart_refctd_ptr<const IGPUImage>(dstImage)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_noCommands = false;
    return copyBufferToImage_impl(srcBuffer, dstImage, dstImageLayout, regionCount, pRegions);
}

bool IGPUCommandBuffer::copyImageToBuffer(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, const IGPUBuffer* const dstBuffer, const uint32_t regionCount, const IGPUImage::SBufferCopy* const pRegions)
{
    if (regionCount==0u)
    {
        NBL_LOG_ERROR("regionCount must be larger than 0!");
        return false;
    }
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT|queue_flags_t::TRANSFER_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;
    
    if (invalidSourceImage(srcImage,srcImageLayout))
        return false;
    if (invalidBuffer(dstBuffer,IGPUBuffer::EUF_TRANSFER_DST_BIT))
        return false;

    // pRegions is too expensive to validate

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyImageToBufferCmd>(m_commandList, core::smart_refctd_ptr<const IGPUImage>(srcImage), core::smart_refctd_ptr<const IGPUBuffer>(dstBuffer)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_noCommands = false;
    return copyImageToBuffer_impl(srcImage, srcImageLayout, dstBuffer, regionCount, pRegions);
}

bool IGPUCommandBuffer::copyImage(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const IGPUImage::SImageCopy* const pRegions)
{
    if (regionCount==0u)
    {
        NBL_LOG_ERROR("regionCount must be larger than 0!");
        return false;
    }
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT|queue_flags_t::TRANSFER_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (invalidSourceImage(srcImage,srcImageLayout))
        return false;
    if (invalidDestinationImage(dstImage,dstImageLayout))
        return false;

    const auto& srcParams = srcImage->getCreationParameters();
    const auto& dstParams = dstImage->getCreationParameters();
    if (srcParams.samples!=dstParams.samples)
    {
        NBL_LOG_ERROR("source and destination have unequal sample count!");
        return false;
    }
    if (asset::getBytesPerPixel(srcParams.format)!=asset::getBytesPerPixel(dstParams.format))
    {
        NBL_LOG_ERROR("source and destination have unequal pixel strides!");
        return false;
    }

    // pRegions is too expensive to validate
    if (!dstImage->validateCopies(pRegions,pRegions+regionCount,srcImage))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyImageCmd>(m_commandList, core::smart_refctd_ptr<const IGPUImage>(srcImage), core::smart_refctd_ptr<const IGPUImage>(dstImage)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_noCommands = false;
    return copyImage_impl(srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);
}

bool IGPUCommandBuffer::invalidShaderGroups(
    const asset::SBufferRange<const IGPUBuffer>& raygenGroupRange, uint32_t raygenGroupStride,
    const asset::SBufferRange<const IGPUBuffer>& missGroupsRange, uint32_t missGroupStride,
    const asset::SBufferRange<const IGPUBuffer>& hitGroupsRange, uint32_t hitGroupStride,
    const asset::SBufferRange<const IGPUBuffer>& callableGroupsRange, uint32_t callableGroupStride, 
    core::bitflag<IGPURayTracingPipeline::SCreationParams::FLAGS> flags) const
{

    using PipelineFlag = IGPURayTracingPipeline::SCreationParams::FLAGS;
    using PipelineFlags = core::bitflag<PipelineFlag>;

    // https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdTraceRaysKHR.html#VUID-vkCmdTraceRaysKHR-flags-03696
    // https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdTraceRaysKHR.html#VUID-vkCmdTraceRaysKHR-flags-03697
    // https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdTraceRaysKHR.html#VUID-vkCmdTraceRaysKHR-flags-03512
    const auto shouldHaveHitGroup = flags & 
      (PipelineFlags(PipelineFlag::RAY_TRACING_NO_NULL_ANY_HIT_SHADERS_BIT_KHR) | 
        PipelineFlag::RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS_BIT_KHR |
        PipelineFlag::RAY_TRACING_NO_NULL_INTERSECTION_SHADERS_BIT_KHR);
    if (shouldHaveHitGroup && !hitGroupsRange.buffer)
    {
        NBL_LOG_ERROR("bound pipeline indicates that traceRays command should have hit group, but hitGroupsRange.buffer is null!");
        return true;
    }

    // https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdTraceRaysKHR.html#VUID-vkCmdTraceRaysKHR-flags-03511
    const auto shouldHaveMissGroup = flags & PipelineFlag::RAY_TRACING_NO_NULL_MISS_SHADERS_BIT_KHR;
    if (shouldHaveMissGroup && !missGroupsRange.buffer)
    {
        NBL_LOG_ERROR("bound pipeline indicates that traceRays command should have hit group, but hitGroupsRange.buffer is null!");
        return true;
    }

    const auto& limits = getOriginDevice()->getPhysicalDevice()->getLimits();
    auto invalidBufferRegion = [this, &limits](const asset::SBufferRange<const IGPUBuffer>& range, uint32_t stride, const char* groupName) -> bool
    {
        const IGPUBuffer* const buffer = range.buffer.get();

        if (!buffer) return false;

        if (!range.isValid())
        {
            NBL_LOG_ERROR("%s buffer range is not valid!", groupName);
            return true;
        }

        if (!(buffer->getCreationParams().usage & IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT))
        {
            NBL_LOG_ERROR("%s buffer must have EUF_SHADER_DEVICE_ADDRESS_BIT usage!", groupName);
            return true;
        }

        // https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdTraceRaysKHR.html#VUID-vkCmdTraceRaysKHR-pRayGenShaderBindingTable-03689
        if (range.offset % limits.shaderGroupBaseAlignment != 0)
        {
            NBL_LOG_ERROR("%s buffer offset must be multiple of %u!", limits.shaderGroupBaseAlignment);
            return true;
        }

        // https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdTraceRaysKHR.html#VUID-vkCmdTraceRaysKHR-pHitShaderBindingTable-03690
        if (stride % limits.shaderGroupHandleAlignment)
        {
            NBL_LOG_ERROR("%s buffer offset must be multiple of %u!", limits.shaderGroupHandleAlignment);
            return true;
        }

        // https://registry.khronos.org/vulkan/specs/latest/man/html/vkCmdTraceRaysKHR.html#VUID-vkCmdTraceRaysKHR-pRayGenShaderBindingTable-03681
        if (!(buffer->getCreationParams().usage & IGPUBuffer::EUF_SHADER_BINDING_TABLE_BIT))
        {
            NBL_LOG_ERROR("%s buffer must have EUF_SHADER_BINDING_TABLE_BIT usage!", groupName);
            return true;
        }

        return false;
    };

    if (invalidBufferRegion(raygenGroupRange, raygenGroupStride, "Raygen Group")) return true;
    if (invalidBufferRegion(missGroupsRange, missGroupStride, "Miss groups")) return true;
    if (invalidBufferRegion(hitGroupsRange, hitGroupStride, "Hit groups")) return true;
    if (invalidBufferRegion(callableGroupsRange, callableGroupStride, "Callable groups")) return true;
    return false;
}

template<class DeviceBuildInfo, typename BuildRangeInfos>
uint32_t IGPUCommandBuffer::buildAccelerationStructures_common(const std::span<const DeviceBuildInfo> infos, BuildRangeInfos ranges, const IGPUBuffer* const indirectBuffer)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    const auto& features = getOriginDevice()->getEnabledFeatures();
    if (!features.accelerationStructure)
    {
        NBL_LOG_ERROR("'accelerationStructure' feature not enabled!");
        return false;
    }

    uint32_t totalGeometries = 0u;
    uint32_t resourcesToTrack = 0u;
    for (auto i=0u; i<infos.size(); i++)
    {
        // valid also checks that the `ranges` are below device limits
        const auto toAdd = infos[i].valid(ranges[i]);
        if (toAdd==0)
        {
            NBL_LOG_ERROR("Acceleration Structure Build Info combined with Range %d is not valid!",i);
            return false;
        }
        if (!isCompatibleDevicewise(infos[i].dstAS))
        {
            NBL_LOG_ERROR(
                "Acceleration Structure Build Info %d destination acceleration structure `%s` is not compatible with our Logical Device!",
                i,infos[i].dstAS ? infos[i].dstAS->getObjectDebugName():"nullptr"
            );
            return false;
        }
        resourcesToTrack += toAdd;
        totalGeometries += infos[i].inputCount();
    }
    // infos array was empty
    if (resourcesToTrack==0u)
    {
        NBL_LOG_ERROR("Acceleration Structure Build Info span was empty!");
        return false;
    }

    if (indirectBuffer)
    {
        if (!features.accelerationStructureIndirectBuild)
        {
            NBL_LOG_ERROR("'accelerationStructureIndirectBuild' feature not enabled!");
            return false;
        }
        resourcesToTrack++;
    }
            
    auto cmd = m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBuildAccelerationStructuresCmd>(m_commandList,resourcesToTrack);
    if (!cmd)
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    auto oit = cmd->getVariableCountResources();
    if (indirectBuffer)
        *(oit++) = core::smart_refctd_ptr<const IGPUBuffer>(indirectBuffer);
    for (const auto& info : infos)
        oit = info.fillTracking(oit);

    return totalGeometries;
}
template uint32_t IGPUCommandBuffer::buildAccelerationStructures_common<IGPUBottomLevelAccelerationStructure::DeviceBuildInfo, IGPUBottomLevelAccelerationStructure::DirectBuildRangeRangeInfos>(
    const std::span<const IGPUBottomLevelAccelerationStructure::DeviceBuildInfo>, IGPUBottomLevelAccelerationStructure::DirectBuildRangeRangeInfos, const IGPUBuffer* const
);
template uint32_t IGPUCommandBuffer::buildAccelerationStructures_common<IGPUBottomLevelAccelerationStructure::DeviceBuildInfo, IGPUBottomLevelAccelerationStructure::MaxInputCounts* const>(
    const std::span<const IGPUBottomLevelAccelerationStructure::DeviceBuildInfo>, IGPUBottomLevelAccelerationStructure::MaxInputCounts* const, const IGPUBuffer* const
);
template uint32_t IGPUCommandBuffer::buildAccelerationStructures_common<IGPUTopLevelAccelerationStructure::DeviceBuildInfo, IGPUTopLevelAccelerationStructure::DirectBuildRangeRangeInfos>(
    const std::span<const IGPUTopLevelAccelerationStructure::DeviceBuildInfo>, IGPUTopLevelAccelerationStructure::DirectBuildRangeRangeInfos, const IGPUBuffer* const
);
template uint32_t IGPUCommandBuffer::buildAccelerationStructures_common<IGPUTopLevelAccelerationStructure::DeviceBuildInfo, IGPUTopLevelAccelerationStructure::MaxInputCounts* const>(
    const std::span<const IGPUTopLevelAccelerationStructure::DeviceBuildInfo>, IGPUTopLevelAccelerationStructure::MaxInputCounts* const, const IGPUBuffer* const
);


bool IGPUCommandBuffer::copyAccelerationStructure(const IGPUAccelerationStructure::CopyInfo& copyInfo)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::TRANSFER_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (!copyInfo.src || !this->isCompatibleDevicewise(copyInfo.src))
    {
        NBL_LOG_ERROR("invalid source copy info!");
        return false;
    }
    if (!copyInfo.dst || !this->isCompatibleDevicewise(copyInfo.dst))
    {
        NBL_LOG_ERROR("invalid destination copy info!");
        return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyAccelerationStructureCmd>(m_commandList, core::smart_refctd_ptr<const IGPUAccelerationStructure>(copyInfo.src), core::smart_refctd_ptr<const IGPUAccelerationStructure>(copyInfo.dst)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_noCommands = false;
    return copyAccelerationStructure_impl(copyInfo);
}

bool IGPUCommandBuffer::copyAccelerationStructureToMemory(const IGPUAccelerationStructure::DeviceCopyToMemoryInfo& copyInfo)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::TRANSFER_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (!copyInfo.src || !this->isCompatibleDevicewise(copyInfo.src))
    {
        NBL_LOG_ERROR("invalid source copy info!");
        return false;
    }
    if (invalidBufferBinding(copyInfo.dst,256u,IGPUBuffer::EUF_TRANSFER_DST_BIT))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyAccelerationStructureToOrFromMemoryCmd>(m_commandList, core::smart_refctd_ptr<const IGPUAccelerationStructure>(copyInfo.src), core::smart_refctd_ptr<const IGPUBuffer>(copyInfo.dst.buffer)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_noCommands = false;
    return copyAccelerationStructureToMemory_impl(copyInfo);
}

bool IGPUCommandBuffer::copyAccelerationStructureFromMemory(const IGPUAccelerationStructure::DeviceCopyFromMemoryInfo& copyInfo)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::TRANSFER_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;
    
    if (invalidBufferBinding(copyInfo.src,256u,IGPUBuffer::EUF_TRANSFER_SRC_BIT))
        return false;
    if (!copyInfo.dst || !this->isCompatibleDevicewise(copyInfo.dst))
    {
        NBL_LOG_ERROR("invalid destination copy info!");
        return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyAccelerationStructureToOrFromMemoryCmd>(m_commandList, core::smart_refctd_ptr<const IGPUAccelerationStructure>(copyInfo.dst), core::smart_refctd_ptr<const IGPUBuffer>(copyInfo.src.buffer)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_noCommands = false;
    return copyAccelerationStructureFromMemory_impl(copyInfo);
}


bool IGPUCommandBuffer::bindComputePipeline(const IGPUComputePipeline* const pipeline)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT))
        return false;

    if (!this->isCompatibleDevicewise(pipeline))
    {
        NBL_LOG_ERROR("incompatible pipeline device!");
        return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBindComputePipelineCmd>(m_commandList, core::smart_refctd_ptr<const IGPUComputePipeline>(pipeline)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_boundComputePipeline = pipeline;

    m_noCommands = false;
    bindComputePipeline_impl(pipeline);

    return true;
}

bool IGPUCommandBuffer::bindGraphicsPipeline(const IGPUGraphicsPipeline* const pipeline)
{
    // Because binding of the Gfx pipeline can happen outside of a Renderpass Scope,
    // we cannot check renderpass-pipeline compatibility here.
    // And checking before every drawcall would be performance suicide.
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT))
        return false;

    if (!pipeline || !this->isCompatibleDevicewise(pipeline))
    {
        NBL_LOG_ERROR("incompatible pipeline device!");
        return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBindGraphicsPipelineCmd>(m_commandList, core::smart_refctd_ptr<const IGPUGraphicsPipeline>(pipeline)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_boundGraphicsPipeline = pipeline;

    m_noCommands = false;
    return bindGraphicsPipeline_impl(pipeline);
}

bool IGPUCommandBuffer::bindRayTracingPipeline(const IGPURayTracingPipeline* const pipeline)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT))
        return false;

    if (!pipeline || !this->isCompatibleDevicewise(pipeline))
    {
        NBL_LOG_ERROR("incompatible pipeline device!");
        return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBindRayTracingPipelineCmd>(m_commandList, core::smart_refctd_ptr<const IGPURayTracingPipeline>(pipeline)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_boundRayTracingPipeline = pipeline;

    m_noCommands = false;
    return bindRayTracingPipeline_impl(pipeline);
}

bool IGPUCommandBuffer::bindDescriptorSets(
    const asset::E_PIPELINE_BIND_POINT pipelineBindPoint, const IGPUPipelineLayout* const layout,
    const uint32_t firstSet, const uint32_t descriptorSetCount, const IGPUDescriptorSet* const* const pDescriptorSets,
    const uint32_t dynamicOffsetCount, const uint32_t* const dynamicOffsets)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT))
        return false;

    if (!layout ||!this->isCompatibleDevicewise(layout))
    {
        NBL_LOG_ERROR("invalid layout!");
        return false;
    }

    for (uint32_t i=0u; i<descriptorSetCount; ++i)
    if (pDescriptorSets[i])
    {
        if (!this->isCompatibleDevicewise(pDescriptorSets[i]))
        {
            NBL_LOG_ERROR("pDescriptorSets[%d] was not created by the same ILogicalDevice as the commandbuffer!", i);
            return false;
        }
        if (!pDescriptorSets[i]->getLayout()->isIdenticallyDefined(layout->getDescriptorSetLayout(firstSet + i)))
        {
            NBL_LOG_ERROR("pDescriptorSets[%d] not identically defined as layout's %dth descriptor layout!", i, firstSet+i);
            return false;
        }
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBindDescriptorSetsCmd>(m_commandList,core::smart_refctd_ptr<const IGPUPipelineLayout>(layout),descriptorSetCount,pDescriptorSets))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    for (uint32_t i=0u; i<descriptorSetCount; ++i)
    if (pDescriptorSets[i] && pDescriptorSets[i]->getLayout()->versionChangeInvalidatesCommandBuffer())
    {
        const auto currentVersion = pDescriptorSets[i]->getVersion();

        auto found = m_boundDescriptorSetsRecord.find(pDescriptorSets[i]);

        if (found != m_boundDescriptorSetsRecord.end())
        {
            if (found->second != currentVersion)
            {
                const char* debugName = pDescriptorSets[i]->getDebugName();
                if (debugName)
                    NBL_LOG_ERROR("Descriptor set (%s, %p) was modified between two recorded bind commands since the last command buffer's beginning!", debugName, pDescriptorSets[i])
                else
                    NBL_LOG_ERROR("Descriptor set (%p)  was modified between two recorded bind commands since the last command buffer's beginning!", pDescriptorSets[i]);

                m_state = STATE::INVALID;
                return false;
            }
        }
        else
            m_boundDescriptorSetsRecord.insert({ pDescriptorSets[i], currentVersion });
    }

    m_noCommands = false;
    return bindDescriptorSets_impl(pipelineBindPoint, layout, firstSet, descriptorSetCount, pDescriptorSets, dynamicOffsetCount, dynamicOffsets);
}

bool IGPUCommandBuffer::pushConstants(const IGPUPipelineLayout* const layout, const core::bitflag<IGPUShader::E_SHADER_STAGE> stageFlags, const uint32_t offset, const uint32_t size, const void* const pValues)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT))
        return false;

    if (!layout || !this->isCompatibleDevicewise(layout))
    {
        NBL_LOG_ERROR("invalid layout!");
        return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CPushConstantsCmd>(m_commandList, core::smart_refctd_ptr<const IGPUPipelineLayout>(layout)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_noCommands = false;
    return pushConstants_impl(layout, stageFlags, offset, size, pValues);
}

bool IGPUCommandBuffer::bindVertexBuffers(const uint32_t firstBinding, const uint32_t bindingCount, const asset::SBufferBinding<const IGPUBuffer>* const pBindings)
{
    if (firstBinding+bindingCount>asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT)
    {
        NBL_LOG_ERROR("bindings count exceeded the maximum allowed bindings!");
        return false;
    }

    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT))
        return false;

    for (uint32_t i=0u; i<bindingCount; ++i)
    if (pBindings[i].buffer && invalidBufferBinding(pBindings[i],4u/*or should we derive from component format?*/,IGPUBuffer::EUF_VERTEX_BUFFER_BIT))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBindVertexBuffersCmd>(m_commandList,bindingCount,pBindings))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_noCommands = false;
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
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_noCommands = false;
    return bindIndexBuffer_impl(binding,indexType);
}


bool IGPUCommandBuffer::invalidDynamic(const uint32_t first, const uint32_t count)
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT))
        return true;

    if (count==0u)
        return true;

    if (first+count>getOriginDevice()->getPhysicalDevice()->getLimits().maxViewports)
        return true;

    return false;
}

bool IGPUCommandBuffer::setLineWidth(const float width)
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT))
        return false;

    const auto device = getOriginDevice();
    if (device->getEnabledFeatures().wideLines)
    {
        const auto& limits = device->getPhysicalDevice()->getLimits();
        if (width<limits.lineWidthRange[0] || width>limits.lineWidthRange[1])
        {
            NBL_LOG_ERROR("width(%d) is out of the allowable range [%d, %d]!", width, limits.lineWidthRange[0], limits.lineWidthRange[1]);
            return false;
        }
    }
    else if (width!=1.f)
    {
        NBL_LOG_ERROR("invalid width(%d). only 1.0 is supported!", width);
        return false;
    }

    m_noCommands = false;
    return setLineWidth_impl(width);
}

bool IGPUCommandBuffer::setDepthBounds(const float minDepthBounds, const float maxDepthBounds)
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT))
        return false;

    if (!getOriginDevice()->getEnabledFeatures().depthBounds)
    {
        NBL_LOG_ERROR("feature not enabled!");
        return false;
    }
    // TODO: implement and handle VK_EXT_depth_range_unrestrices
    if (minDepthBounds<0.f || maxDepthBounds>1.f)
    {
        NBL_LOG_ERROR("invalid bounds [%d, %d]!", minDepthBounds, maxDepthBounds);
        return false;
    }

    m_noCommands = false;
    return setDepthBounds_impl(minDepthBounds,maxDepthBounds);
}


bool IGPUCommandBuffer::resetQueryPool(IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount)
{
    // also encode/decode and opticalflow (video) queues
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (!queryPool || !this->isCompatibleDevicewise(queryPool))
    {
        NBL_LOG_ERROR("invalid parameter 'queryPool'!");
        return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CResetQueryPoolCmd>(m_commandList, core::smart_refctd_ptr<const IQueryPool>(queryPool)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_noCommands = false;
    return resetQueryPool_impl(queryPool, firstQuery, queryCount);
}

bool IGPUCommandBuffer::beginQuery(IQueryPool* const queryPool, const uint32_t query, const core::bitflag<QUERY_CONTROL_FLAGS> flags)
{
    // also encode/decode and opticalflow (video) queues
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT))
        return false;

    if (!queryPool || !this->isCompatibleDevicewise(queryPool))
    {
        NBL_LOG_ERROR("invalid parameter 'queryPool'!");
        return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBeginQueryCmd>(m_commandList, core::smart_refctd_ptr<const IQueryPool>(queryPool)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_noCommands = false;
    return beginQuery_impl(queryPool, query, flags);
}

bool IGPUCommandBuffer::endQuery(IQueryPool* const queryPool, const uint32_t query)
{
    // also encode/decode and opticalflow (video) queues
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT))
        return false;

    if (!queryPool || !this->isCompatibleDevicewise(queryPool))
    {
        NBL_LOG_ERROR("invalid parameter 'queryPool'!");
        return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CEndQueryCmd>(m_commandList, core::smart_refctd_ptr<const IQueryPool>(queryPool)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_noCommands = false;
    return endQuery_impl(queryPool, query);
}

bool IGPUCommandBuffer::writeTimestamp(const stage_flags_t pipelineStage, IQueryPool* const queryPool, const uint32_t query)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT|queue_flags_t::TRANSFER_BIT))
        return false;

    const auto qFamIx = m_cmdpool->getQueueFamilyIndex();
    if (!getOriginDevice()->getSupportedStageMask(qFamIx).hasFlags(pipelineStage))
    {
        NBL_LOG_ERROR("incompatible parameter 'pipelineStage'!");
        return false;
    }
    if (getOriginDevice()->getPhysicalDevice()->getQueueFamilyProperties()[qFamIx].timestampValidBits == 0u)
    {
        NBL_LOG_ERROR("timestamps not supported for this queue family index (%d)!", qFamIx);
        return false;
    }

    if (!queryPool || !this->isCompatibleDevicewise(queryPool) || queryPool->getCreationParameters().queryType!=IQueryPool::TYPE::TIMESTAMP || query>=queryPool->getCreationParameters().queryCount)
    {
        NBL_LOG_ERROR("invalid parameter 'queryPool'!");
        return false;
    }

    assert(core::isPoT(static_cast<uint32_t>(pipelineStage))); // should only be 1 stage (1 bit set)

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CWriteTimestampCmd>(m_commandList, core::smart_refctd_ptr<const IQueryPool>(queryPool)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_noCommands = false;
    return writeTimestamp_impl(pipelineStage, queryPool, query);
}

bool IGPUCommandBuffer::writeAccelerationStructureProperties(const std::span<const IGPUAccelerationStructure* const> pAccelerationStructures, const IQueryPool::TYPE queryType, IQueryPool* const queryPool, const uint32_t firstQuery)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (!queryPool || !this->isCompatibleDevicewise(queryPool))
    {
        NBL_LOG_ERROR("invalid parameter 'queryPool'!");
        return false;
    }

    if (pAccelerationStructures.empty())
    {
        NBL_LOG_ERROR("parameter 'pAccelerationStructures' is empty!");
        return false;
    }

    for (auto& as : pAccelerationStructures)
    {
        if (!isCompatibleDevicewise(as))
        {
            NBL_LOG_ERROR("incompatible device!");
            return false;
        }
    }

    auto cmd = m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CWriteAccelerationStructurePropertiesCmd>(m_commandList, queryPool, pAccelerationStructures.size());
    if (!cmd)
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    auto oit = cmd->getVariableCountResources();
    for (auto& as : pAccelerationStructures)
        *(oit++) = core::smart_refctd_ptr<const core::IReferenceCounted>(as);
    m_noCommands = false;
    return writeAccelerationStructureProperties_impl(pAccelerationStructures, queryType, queryPool, firstQuery);
}

bool IGPUCommandBuffer::copyQueryPoolResults(
    const IQueryPool* const queryPool, const uint32_t firstQuery, const uint32_t queryCount,
    const asset::SBufferBinding<IGPUBuffer>& dstBuffer, const size_t stride, const core::bitflag<IQueryPool::RESULTS_FLAGS> flags)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    // TODO: rest of validation
    if (!queryPool || !this->isCompatibleDevicewise(queryPool))
    {
        NBL_LOG_ERROR("invalid parameter 'queryPool'!");
        return false;
    }
    if (queryCount==0u || firstQuery+queryCount>=queryPool->getCreationParameters().queryCount)
    {
        NBL_LOG_ERROR("parameter 'queryCount' exceeded the valid range [1, %d]!", queryPool->getCreationParameters().queryCount - firstQuery);
        return false;
    }

    const size_t alignment = flags.hasFlags(IQueryPool::RESULTS_FLAGS::_64_BIT) ? alignof(uint64_t):alignof(uint32_t);
    if (invalidBufferRange({dstBuffer.offset,queryCount*stride,dstBuffer.buffer},alignment,IGPUBuffer::EUF_TRANSFER_DST_BIT))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CCopyQueryPoolResultsCmd>(m_commandList, core::smart_refctd_ptr<const IQueryPool>(queryPool), core::smart_refctd_ptr<const IGPUBuffer>(dstBuffer.buffer)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_noCommands = false;
    return copyQueryPoolResults_impl(queryPool, firstQuery, queryCount, dstBuffer, stride, flags);
}


bool IGPUCommandBuffer::dispatch(const uint32_t groupCountX, const uint32_t groupCountY, const uint32_t groupCountZ)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (groupCountX==0 || groupCountY==0 || groupCountZ==0)
    {
        NBL_LOG_ERROR("invalid group counts (%d, %d, %d)!", groupCountX, groupCountY, groupCountZ);
        return false;
    }

    const auto& limits = getOriginDevice()->getPhysicalDevice()->getLimits();
    if (groupCountX>limits.maxComputeWorkGroupCount[0] || groupCountY>limits.maxComputeWorkGroupCount[1] || groupCountZ>limits.maxComputeWorkGroupCount[2])
    {
        NBL_LOG_ERROR("group counts (%d, %d, %d) exceeds maximum counts (%d, %d, %d)!", groupCountX, groupCountY, groupCountZ, limits.maxComputeWorkGroupCount[0], limits.maxComputeWorkGroupCount[1], limits.maxComputeWorkGroupCount[2]);
        return false;
    }

    m_noCommands = false;
    return dispatch_impl(groupCountX,groupCountY,groupCountZ);
}

bool IGPUCommandBuffer::dispatchIndirect(const asset::SBufferBinding<const IGPUBuffer>& binding)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (invalidBufferBinding(binding,4u/*TODO: is it really 4?*/,IGPUBuffer::EUF_INDIRECT_BUFFER_BIT))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CIndirectCmd>(m_commandList,core::smart_refctd_ptr<const IGPUBuffer>(binding.buffer)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_noCommands = false;
    return dispatchIndirect_impl(binding);
}


bool IGPUCommandBuffer::beginRenderPass(SRenderpassBeginInfo info, const SUBPASS_CONTENTS contents)
{
    if (m_recordingFlags.hasFlags(USAGE::RENDER_PASS_CONTINUE_BIT))
    {
        NBL_LOG_ERROR("primary command buffer must not include the RENDER_PASS_CONTINUE_BIT flag!");
        return false;
    }
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdBeginRenderPass2KHR.html#VUID-vkCmdBeginRenderPass2-framebuffer-02779
    if (!info.framebuffer || !this->isCompatibleDevicewise(info.framebuffer))
    {
        NBL_LOG_ERROR("invalid framebuffer!");
        return false;
    }

    const auto& renderArea = info.renderArea;
    if (renderArea.extent.width==0u || renderArea.extent.height==0u)
    {
        NBL_LOG_ERROR("invalid extent size [%d, %d]!", renderArea.extent.width, renderArea.extent.height);
        return false;
    }

    const auto& framebufferParams = info.framebuffer->getCreationParameters();
    if (renderArea.offset.x+renderArea.extent.width>framebufferParams.width || renderArea.offset.y+renderArea.extent.height>framebufferParams.height)
    {
        NBL_LOG_ERROR("render area [%d, %d] exceeds valid range [%d, %d]!",
            renderArea.offset.x + renderArea.extent.width, renderArea.offset.y + renderArea.extent.height,
            framebufferParams.width, framebufferParams.height);
        return false;
    }

    if (info.renderpass)
    {
        if (!framebufferParams.renderpass->compatible(info.renderpass))
        {
            NBL_LOG_ERROR("renderpass is incompatible with the framebuffer!");
            return false;
        }
    }
    else
        info.renderpass = framebufferParams.renderpass.get();

    if (info.renderpass->getDepthStencilLoadOpAttachmentEnd()!=0u && !info.depthStencilClearValues)
    {
        NBL_LOG_ERROR("depthStencilClearValues must be greater than the largest attachment index specifying a load Op of CLEAR!");
        return false;
    }
    if (info.renderpass->getColorLoadOpAttachmentEnd()!=0u && !info.colorClearValues)
    {
        NBL_LOG_ERROR("colorClearValues must be greater than the largest attachment index specifying a load Op of CLEAR!");
        return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBeginRenderPassCmd>(m_commandList,core::smart_refctd_ptr<const IGPURenderpass>(info.renderpass),core::smart_refctd_ptr<const IGPUFramebuffer>(info.framebuffer)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_noCommands = false;
    if (!beginRenderPass_impl(info,contents))
        return false;
    m_cachedInheritanceInfo.renderpass = info.renderpass;
    m_cachedInheritanceInfo.subpass = 0;
    m_cachedInheritanceInfo.framebuffer = info.framebuffer;
    return true;
}

bool IGPUCommandBuffer::nextSubpass(const SUBPASS_CONTENTS contents)
{
    if (m_recordingFlags.hasFlags(USAGE::RENDER_PASS_CONTINUE_BIT))
    {
        NBL_LOG_ERROR("primary command buffer must not include the RENDER_PASS_CONTINUE_BIT flag!");
        return false;
    }
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdNextSubpass2KHR.html#VUID-vkCmdNextSubpass2-None-03102
    if (m_cachedInheritanceInfo.subpass+1>=m_cachedInheritanceInfo.renderpass->getSubpassCount())
    {
        NBL_LOG_ERROR("no more subpasses to transit!");
        return false;
    }

    m_cachedInheritanceInfo.subpass++;
    m_noCommands = false;
    return nextSubpass_impl(contents);
}

bool IGPUCommandBuffer::endRenderPass()
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::INSIDE))
        return false;
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdEndRenderPass2KHR.html#VUID-vkCmdEndRenderPass2-None-03103
    if (m_cachedInheritanceInfo.subpass+1!=m_cachedInheritanceInfo.renderpass->getSubpassCount())
    {
        NBL_LOG_ERROR("the amount of transited (%d) sub-passes must be equal to total sub-pass count!", m_cachedInheritanceInfo.subpass + 1, m_cachedInheritanceInfo.renderpass->getSubpassCount());
        return false;
    }

    m_cachedInheritanceInfo.subpass = SInheritanceInfo{}.subpass;
    m_noCommands = false;
    return endRenderPass_impl();
}


bool IGPUCommandBuffer::clearAttachments(const SClearAttachments& info)
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::INSIDE))
        return false;

    if (!info.valid())
    {
        NBL_LOG_ERROR("invalid parameter 'info'!");
        return false;
    }

    const auto& rpassParams = m_cachedInheritanceInfo.renderpass->getCreationParameters();
    const auto& subpass = rpassParams.subpasses[m_cachedInheritanceInfo.subpass];
    if (info.clearDepth||info.clearStencil)
    {
        if (!subpass.depthStencilAttachment.render.used())
        {
            NBL_LOG_ERROR("current subpass attachment and the clear format doesn't match!");
            return false;
        }
        const auto& depthStencilAttachment = rpassParams.depthStencilAttachments[subpass.depthStencilAttachment.render.attachmentIndex];
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdClearAttachments.html#VUID-vkCmdClearAttachments-aspectMask-07884
        if (info.clearDepth && asset::isStencilOnlyFormat(depthStencilAttachment.format))
        {
            NBL_LOG_ERROR("stencil only asset can't be cleared with the 'clearDepth' parameter!");
            return false;
        }
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdClearAttachments.html#VUID-vkCmdClearAttachments-aspectMask-07885
        if (info.clearStencil && asset::isDepthOnlyFormat(depthStencilAttachment.format))
        {
            NBL_LOG_ERROR("depth only asset can't be cleared with the 'clearStencil' parameter!");
            return false;
        }
    }
    for (auto i=0; i<sizeof(info.clearColorMask)*8; i++)
    {
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdClearAttachments.html#VUID-vkCmdClearAttachments-aspectMask-07271
        if (info.clearColor(i) && !subpass.colorAttachments[i].render.used())
        {
            NBL_LOG_ERROR("current subpass attachment and the clear format doesn't match!");
            return false;
        }
    }
    
    // cannot validate without tracking more stuff
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdClearAttachments.html#VUID-vkCmdClearAttachments-pRects-00016
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdClearAttachments.html#VUID-vkCmdClearAttachments-pRects-06937
    // TODO: if (m_cachedInheritanceInfo.renderpass->getUsesMultiview()) then, instead check https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdClearAttachments.html#VUID-vkCmdClearAttachments-baseArrayLayer-00018
    for (const SClearAttachments::SRegion& region : info.regions)
    {
        if (region.baseArrayLayer+region.layerCount>m_cachedInheritanceInfo.framebuffer->getCreationParameters().layers)
        {
            NBL_LOG_ERROR("region layers (%d) exceeds the valid amount (%d)!", region.baseArrayLayer + region.layerCount, m_cachedInheritanceInfo.framebuffer->getCreationParameters().layers);
            return false;
        }
    }

    m_noCommands = false;
    return clearAttachments_impl(info);
}


bool IGPUCommandBuffer::draw(const uint32_t vertexCount, const uint32_t instanceCount, const uint32_t firstVertex, const uint32_t firstInstance)
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::INSIDE))
        return false;

    if (vertexCount==0u || instanceCount == 0u)
    {
        NBL_LOG_ERROR("invalid 'vertexCount' (%d) or 'instanceCount' (%d)!", vertexCount, instanceCount);
        return false;
    }

    m_noCommands = false;
    return draw_impl(vertexCount,instanceCount,firstVertex,firstInstance);
}

bool IGPUCommandBuffer::drawIndexed(const uint32_t indexCount, const uint32_t instanceCount, const uint32_t firstIndex, const int32_t vertexOffset, const uint32_t firstInstance)
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::INSIDE))
        return false;

    if (indexCount==0u || instanceCount == 0u)
    {
        NBL_LOG_ERROR("invalid 'indexCount' (%d) or 'instanceCount' (%d)!", indexCount, instanceCount);
        return false;
    }

    m_noCommands = false;
    return drawIndexed_impl(indexCount,instanceCount,firstIndex,vertexOffset,firstInstance);
}

template<typename IndirectCommand> requires nbl::is_any_of_v<IndirectCommand,hlsl::DrawArraysIndirectCommand_t,hlsl::DrawElementsIndirectCommand_t>
bool IGPUCommandBuffer::invalidDrawIndirect(const asset::SBufferBinding<const IGPUBuffer>& binding, const uint32_t drawCount, uint32_t stride)
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::INSIDE))
        return true;

    if (drawCount)
    {
        if (drawCount==1u)
            stride = sizeof(IndirectCommand);
        if (stride&0x3u || stride<sizeof(IndirectCommand))
        {
            NBL_LOG_ERROR("invalid command buffer stride (%d)!", stride);
            return true;
        }
        if (drawCount > getOriginDevice()->getPhysicalDevice()->getLimits().maxDrawIndirectCount)
        {
            NBL_LOG_ERROR("draw count (%d) exceeds maximum allowed amount (%d)!", drawCount, getOriginDevice()->getPhysicalDevice()->getLimits().maxDrawIndirectCount);
            return true;
        }
        if (invalidBufferRange({ binding.offset,stride * (drawCount - 1u) + sizeof(IndirectCommand),binding.buffer }, alignof(uint32_t), IGPUBuffer::EUF_INDIRECT_BUFFER_BIT))
            return true;
    }
    return false;
}
template bool IGPUCommandBuffer::invalidDrawIndirect<hlsl::DrawArraysIndirectCommand_t>(const asset::SBufferBinding<const IGPUBuffer>&, const uint32_t, uint32_t);
template bool IGPUCommandBuffer::invalidDrawIndirect<hlsl::DrawElementsIndirectCommand_t>(const asset::SBufferBinding<const IGPUBuffer>&, const uint32_t, uint32_t);

template<typename IndirectCommand> requires nbl::is_any_of_v<IndirectCommand,hlsl::DrawArraysIndirectCommand_t,hlsl::DrawElementsIndirectCommand_t>
bool IGPUCommandBuffer::invalidDrawIndirectCount(const asset::SBufferBinding<const IGPUBuffer>& indirectBinding, const asset::SBufferBinding<const IGPUBuffer>& countBinding, const uint32_t maxDrawCount, const uint32_t stride)
{
    if (!getOriginDevice()->getPhysicalDevice()->getLimits().drawIndirectCount)
    {
        NBL_LOG_ERROR("indirect draws with draw call count are not supported!");
        return true;
    }

    if (invalidDrawIndirect<IndirectCommand>(indirectBinding,maxDrawCount,stride))
        return true;
    if (invalidBufferRange({countBinding.offset,sizeof(uint32_t),countBinding.buffer},alignof(uint32_t),IGPUBuffer::EUF_INDIRECT_BUFFER_BIT))
        return true;

    return false;
}
template bool IGPUCommandBuffer::invalidDrawIndirectCount<hlsl::DrawArraysIndirectCommand_t>(const asset::SBufferBinding<const IGPUBuffer>&, const asset::SBufferBinding<const IGPUBuffer>&, const uint32_t, const uint32_t);
template bool IGPUCommandBuffer::invalidDrawIndirectCount<hlsl::DrawElementsIndirectCommand_t>(const asset::SBufferBinding<const IGPUBuffer>&, const asset::SBufferBinding<const IGPUBuffer>&, const uint32_t, const uint32_t);

bool IGPUCommandBuffer::drawIndirect(const asset::SBufferBinding<const IGPUBuffer>& binding, const uint32_t drawCount, const uint32_t stride)
{
    if (invalidDrawIndirect<hlsl::DrawArraysIndirectCommand_t>(binding,drawCount,stride))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CIndirectCmd>(m_commandList,core::smart_refctd_ptr<const IGPUBuffer>(binding.buffer)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_noCommands = false;
    return drawIndirect_impl(binding, drawCount, stride);
}

bool IGPUCommandBuffer::drawIndexedIndirect(const asset::SBufferBinding<const IGPUBuffer>& binding, const uint32_t drawCount, const uint32_t stride)
{
    if (invalidDrawIndirect<hlsl::DrawElementsIndirectCommand_t>(binding,drawCount,stride))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CIndirectCmd>(m_commandList, core::smart_refctd_ptr<const IGPUBuffer>(binding.buffer)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_noCommands = false;
    return drawIndexedIndirect_impl(binding, drawCount, stride);
}

bool IGPUCommandBuffer::drawIndirectCount(const asset::SBufferBinding<const IGPUBuffer>& indirectBinding, const asset::SBufferBinding<const IGPUBuffer>& countBinding, const uint32_t maxDrawCount, const uint32_t stride)
{
    if (!invalidDrawIndirectCount<hlsl::DrawArraysIndirectCommand_t>(indirectBinding,countBinding,maxDrawCount,stride))
        return false;

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CDrawIndirectCountCmd>(m_commandList, core::smart_refctd_ptr<const IGPUBuffer>(indirectBinding.buffer), core::smart_refctd_ptr<const IGPUBuffer>(countBinding.buffer)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_noCommands = false;
    return drawIndirectCount_impl(indirectBinding, countBinding, maxDrawCount, stride);
}

bool IGPUCommandBuffer::drawIndexedIndirectCount(const asset::SBufferBinding<const IGPUBuffer>& indirectBinding, const asset::SBufferBinding<const IGPUBuffer>& countBinding, const uint32_t maxDrawCount, const uint32_t stride)
{
    if (!invalidDrawIndirectCount<hlsl::DrawElementsIndirectCommand_t>(indirectBinding,countBinding,maxDrawCount,stride))
        return false;
    
    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CDrawIndirectCountCmd>(m_commandList, core::smart_refctd_ptr<const IGPUBuffer>(indirectBinding.buffer), core::smart_refctd_ptr<const IGPUBuffer>(countBinding.buffer)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_noCommands = false;
    return drawIndexedIndirectCount_impl(indirectBinding, countBinding, maxDrawCount, stride);
}

/*
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
*/
template<bool dst>
static bool disallowedLayoutForBlitAndResolve(const IGPUImage::LAYOUT layout)
{
    switch (layout)
    {
        case IGPUImage::LAYOUT::GENERAL:
            return false;
        case IGPUImage::LAYOUT::TRANSFER_SRC_OPTIMAL:
            if (!dst)
                return false;
            break;
        case IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL:
            if (dst)
                return false;
            break;
        case IGPUImage::LAYOUT::SHARED_PRESENT:
            return false;
        default:
            break;
    }
    return true;
}

bool IGPUCommandBuffer::blitImage(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const std::span<const SImageBlit> regions, const IGPUSampler::E_TEXTURE_FILTER filter)
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (regions.empty() || disallowedLayoutForBlitAndResolve<false>(srcImageLayout) || disallowedLayoutForBlitAndResolve<true>(dstImageLayout))
    {
        NBL_LOG_ERROR("invalid parameters!");
        return false;
    }

    const auto* physDev = getOriginDevice()->getPhysicalDevice();
    const auto& srcParams = srcImage->getCreationParameters();
    if (!srcImage || !this->isCompatibleDevicewise(srcImage) || !srcParams.usage.hasFlags(IGPUImage::EUF_TRANSFER_SRC_BIT) || !physDev->getImageFormatUsages(srcImage->getTiling())[srcParams.format].blitSrc)
    {
        NBL_LOG_ERROR("invalid source image!");
        return false;
    }

    const auto& dstParams = dstImage->getCreationParameters();
    if (!dstImage || !this->isCompatibleDevicewise(dstImage) || !dstParams.usage.hasFlags(IGPUImage::EUF_TRANSFER_DST_BIT) || !physDev->getImageFormatUsages(dstImage->getTiling())[dstParams.format].blitDst)
    {
        NBL_LOG_ERROR("invalid destination image!");
        return false;
    }

    // TODO rest of: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdBlitImage.html#VUID-vkCmdBlitImage-srcImage-00229

    for (auto region : regions)
    {
        if (region.layerCount==0 || !region.aspectMask)
        {
            NBL_LOG_ERROR("invalid region layerCount (%d) or aspectMask (%d)!", (uint32_t)region.layerCount, (uint32_t)region.aspectMask);
            return false;
        }
        // probably validate the offsets, and extents
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CBlitImageCmd>(m_commandList, core::smart_refctd_ptr<const IGPUImage>(srcImage), core::smart_refctd_ptr<const IGPUImage>(dstImage)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_noCommands = false;
    return blitImage_impl(srcImage, srcImageLayout, dstImage, dstImageLayout, regions, filter);
}

bool IGPUCommandBuffer::resolveImage(const IGPUImage* const srcImage, const IGPUImage::LAYOUT srcImageLayout, IGPUImage* const dstImage, const IGPUImage::LAYOUT dstImageLayout, const uint32_t regionCount, const SImageResolve* pRegions)
{
    if (!checkStateBeforeRecording(queue_flags_t::GRAPHICS_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;
    
    if (regionCount==0u || !pRegions || disallowedLayoutForBlitAndResolve<false>(srcImageLayout) || disallowedLayoutForBlitAndResolve<true>(dstImageLayout))
    {
        NBL_LOG_ERROR("invalid parameters!");
        return false;
    }

    const auto* physDev = getOriginDevice()->getPhysicalDevice();
    const auto& srcParams = srcImage->getCreationParameters();
    if (!srcImage || !this->isCompatibleDevicewise(srcImage) || !srcParams.usage.hasFlags(IGPUImage::EUF_TRANSFER_SRC_BIT))
    {
        NBL_LOG_ERROR("invalid source image!");
        return false;
    }
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdResolveImage.html#VUID-vkCmdResolveImage-srcImage-00258
    if (srcParams.samples == IGPUImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT)
    {
        NBL_LOG_ERROR("source image sample count must be 1!");
        return false;
    }

    const auto& dstParams = dstImage->getCreationParameters();
    if (!dstImage || !this->isCompatibleDevicewise(dstImage) || !dstParams.usage.hasFlags(IGPUImage::EUF_TRANSFER_SRC_BIT) || !physDev->getImageFormatUsages(dstImage->getTiling())[dstParams.format].attachment)
    {
        NBL_LOG_ERROR("invalid destination image!");
        return false;
    }
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdResolveImage.html#VUID-vkCmdResolveImage-dstImage-00259
    if (dstParams.samples!=IGPUImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT)
    {
        NBL_LOG_ERROR("destination image sample count must be 1!");
        return false;
    }

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdResolveImage.html#VUID-vkCmdResolveImage-srcImage-01386
    if (srcParams.format!=dstParams.format)
    {
        NBL_LOG_ERROR("source and destination image formats doesn't match!");
        return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CResolveImageCmd>(m_commandList, core::smart_refctd_ptr<const IGPUImage>(srcImage), core::smart_refctd_ptr<const IGPUImage>(dstImage)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_noCommands = false;
    return resolveImage_impl(srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);
}

bool IGPUCommandBuffer::setRayTracingPipelineStackSize(uint32_t pipelineStackSize)
{
    return setRayTracingPipelineStackSize_impl(pipelineStackSize);
}

bool IGPUCommandBuffer::traceRays(
    const asset::SBufferRange<const IGPUBuffer>& raygenGroupRange, uint32_t raygenGroupStride,
    const asset::SBufferRange<const IGPUBuffer>& missGroupsRange, uint32_t missGroupStride,
    const asset::SBufferRange<const IGPUBuffer>& hitGroupsRange, uint32_t hitGroupStride,
    const asset::SBufferRange<const IGPUBuffer>& callableGroupsRange, uint32_t callableGroupStride,
    uint32_t width, uint32_t height, uint32_t depth)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (width == 0 || height == 0 || depth == 0)
    {
        NBL_LOG_ERROR("invalid work counts (%d, %d, %d)!", width, height, depth);
        return false;
    }

    if (m_boundRayTracingPipeline == nullptr)
    {
        NBL_LOG_ERROR("invalid bound pipeline for traceRays command!");
        return false;
    }
    const auto flags = m_boundRayTracingPipeline->getCreationFlags();

    if (invalidShaderGroups(raygenGroupRange, raygenGroupStride, 
        missGroupsRange, missGroupStride, 
        hitGroupsRange, hitGroupStride, 
        callableGroupsRange, callableGroupStride,
        flags))
    {
        NBL_LOG_ERROR("invalid shader groups for traceRays command!");
        return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CTraceRaysCmd>(m_commandList, 
        core::smart_refctd_ptr<const IGPUBuffer>(raygenGroupRange.buffer),
        core::smart_refctd_ptr<const IGPUBuffer>(missGroupsRange.buffer),
        core::smart_refctd_ptr<const IGPUBuffer>(hitGroupsRange.buffer),
        core::smart_refctd_ptr<const IGPUBuffer>(callableGroupsRange.buffer)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    m_noCommands = false;

    return traceRays_impl(
        raygenGroupRange, raygenGroupStride, 
        missGroupsRange, missGroupStride,
        hitGroupsRange, hitGroupStride,
        callableGroupsRange, callableGroupStride,
        width, height, depth);
}

bool IGPUCommandBuffer::traceRaysIndirect(
    const asset::SBufferRange<const IGPUBuffer>& raygenGroupRange, uint32_t raygenGroupStride,
    const asset::SBufferRange<const IGPUBuffer>& missGroupsRange, uint32_t missGroupStride,
    const asset::SBufferRange<const IGPUBuffer>& hitGroupsRange, uint32_t hitGroupStride,
    const asset::SBufferRange<const IGPUBuffer>& callableGroupsRange, uint32_t callableGroupStride,
    const asset::SBufferBinding<const IGPUBuffer>& indirectBinding)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT,RENDERPASS_SCOPE::OUTSIDE))
        return false;

    if (m_boundRayTracingPipeline == nullptr)
    {
        NBL_LOG_ERROR("invalid bound pipeline for traceRays command!");
        return false;
    }
    const auto flags = m_boundRayTracingPipeline->getCreationFlags();

    if (invalidShaderGroups(raygenGroupRange, raygenGroupStride, 
        missGroupsRange, missGroupStride, 
        hitGroupsRange, hitGroupStride, 
        callableGroupsRange, callableGroupStride,
        flags))
    {
        NBL_LOG_ERROR("invalid shader groups for traceRays command!");
        return false;
    }

    if (!m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CTraceRaysIndirectCmd>(m_commandList, 
        core::smart_refctd_ptr<const IGPUBuffer>(raygenGroupRange.buffer),
        core::smart_refctd_ptr<const IGPUBuffer>(missGroupsRange.buffer),
        core::smart_refctd_ptr<const IGPUBuffer>(hitGroupsRange.buffer),
        core::smart_refctd_ptr<const IGPUBuffer>(callableGroupsRange.buffer),
        core::smart_refctd_ptr<const IGPUBuffer>(indirectBinding.buffer)))
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdTraceRaysIndirectKHR.html#VUID-vkCmdTraceRaysIndirectKHR-indirectDeviceAddress-03634
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdTraceRaysIndirectKHR.html#VUID-vkCmdTraceRaysIndirectKHR-indirectDeviceAddress-03633
    if (invalidBufferBinding(indirectBinding, 4u,IGPUBuffer::EUF_INDIRECT_BUFFER_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT))
        return false;

    m_noCommands = false;

    return traceRaysIndirect_impl(
        raygenGroupRange, raygenGroupStride,
        missGroupsRange, missGroupStride,
        hitGroupsRange, hitGroupStride,
        callableGroupsRange, callableGroupStride,
        indirectBinding);
}

bool IGPUCommandBuffer::executeCommands(const uint32_t count, IGPUCommandBuffer* const* const cmdbufs)
{
    if (!checkStateBeforeRecording(queue_flags_t::COMPUTE_BIT|queue_flags_t::GRAPHICS_BIT|queue_flags_t::TRANSFER_BIT))
        return false;

    for (uint32_t i=0u; i<count; ++i)
    {
        if (!cmdbufs[i] || cmdbufs[i]->getLevel()!=IGPUCommandPool::BUFFER_LEVEL::SECONDARY)
        {
            NBL_LOG_ERROR("cmdbufs[%d] level is not SECONDARY!", i);
            return false;
        }

        if (!this->isCompatibleDevicewise(cmdbufs[i]))
        {
            NBL_LOG_ERROR("cmdbufs[%d] has incompatible device!", i);
            return false;
        }
    }

    auto cmd = m_cmdpool->m_commandListPool.emplace<IGPUCommandPool::CExecuteCommandsCmd>(m_commandList, count);
    if (!cmd)
    {
        NBL_LOG_ERROR("out of host memory!");
        return false;
    }
    for (auto i=0u; i<count; i++)
        cmd->getVariableCountResources()[i] = core::smart_refctd_ptr<const core::IReferenceCounted>(cmdbufs[i]);
    m_noCommands = false;
    return executeCommands_impl(count,cmdbufs);
}

}