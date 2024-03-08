#include "nbl/video/IQueue.h"
#include "nbl/video/IGPUCommandBuffer.h"
#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/TimelineEventHandlers.h"

namespace nbl::video
{

IQueue::IQueue(ILogicalDevice* originDevice, const uint32_t _famIx, const core::bitflag<CREATE_FLAGS> _flags, const float _priority)
    : m_originDevice(originDevice), m_familyIndex(_famIx), m_priority(_priority), m_flags(_flags)
{
    m_submittedResources = std::make_unique<MultiTimelineEventHandlerST<DeferredSubmitResourceDrop,false>>(originDevice);
}

auto IQueue::submit(const std::span<const SSubmitInfo> _submits) -> RESULT
{
    if (_submits.empty())
        return RESULT::OTHER_ERROR;

    auto* logger = m_originDevice->getPhysicalDevice()->getDebugCallback()->getLogger();
    for (const auto& submit : _submits)
    {
        if (!submit.valid())
            return RESULT::OTHER_ERROR;

        auto invalidSemaphores = [this,logger](const std::span<const SSubmitInfo::SSemaphoreInfo> semaphoreInfos) -> bool
        {
            for (const auto& semaphoreInfo : semaphoreInfos)
            {
                auto* sema = semaphoreInfo.semaphore;
                if (!sema || !sema->wasCreatedBy(m_originDevice))
                {
                    logger->log("Why on earth are you trying to submit a nullptr semaphore or to a wrong device!?", system::ILogger::ELL_ERROR);
                    return true;
                }
            }
            return false;
        };
        if (invalidSemaphores(submit.waitSemaphores) || invalidSemaphores(submit.signalSemaphores))
            return RESULT::OTHER_ERROR;

        for (const auto& commandBuffer : submit.commandBuffers)
        {
            auto* cmdbuf = commandBuffer.cmdbuf;
            if (!cmdbuf || !cmdbuf->wasCreatedBy(m_originDevice))
            {
                logger->log("Why on earth are you trying to submit a nullptr command buffer or to a wrong device!?", system::ILogger::ELL_ERROR);
                return RESULT::OTHER_ERROR;
            }

            const char* commandBufferDebugName = cmdbuf->getDebugName();
            if (!commandBufferDebugName)
                commandBufferDebugName = "";

            if (cmdbuf->getLevel()!=IGPUCommandPool::BUFFER_LEVEL::PRIMARY)
            {
                logger->log("Command buffer (%s, %p) is NOT PRIMARY LEVEL", system::ILogger::ELL_ERROR, commandBufferDebugName, cmdbuf);
                return RESULT::OTHER_ERROR;
            }
            if (cmdbuf->getState()!=IGPUCommandBuffer::STATE::EXECUTABLE)
            {
                logger->log("Command buffer (%s, %p) is NOT IN THE EXECUTABLE STATE", system::ILogger::ELL_ERROR, commandBufferDebugName, cmdbuf);
                return RESULT::OTHER_ERROR;
            }

            const auto& descriptorSetsRecord = cmdbuf->getBoundDescriptorSetsRecord();
            for (const auto& dsRecord : descriptorSetsRecord)
            {
                const auto& [ds, cachedDSVersion] = dsRecord;
                if (ds->getVersion() > cachedDSVersion)
                {
                    logger->log("Descriptor set(s) updated after being bound without UPDATE_AFTER_BIND. Invalidating command buffer (%s, %p)..", system::ILogger::ELL_WARNING, commandBufferDebugName, cmdbuf);
                    cmdbuf->m_state = IGPUCommandBuffer::STATE::INVALID;
                    return RESULT::OTHER_ERROR;
                }
            }
        }
    }

    // mark cmdbufs as pending
    for (const auto& submit : _submits)
    for (const auto& commandBuffer : submit.commandBuffers)
        commandBuffer.cmdbuf->m_state = IGPUCommandBuffer::STATE::PENDING;
    // do the submit
    auto result = submit_impl(_submits);
    if (result!=RESULT::SUCCESS)
        return result;
    // poll for whatever is done, free up memory ASAP
    // we poll everything (full GC run) because we would release memory very slowly otherwise
    m_submittedResources->poll();
    //
    for (const auto& submit : _submits)
    {
        // hold onto the semaphores and commandbuffers until the submit signals the last semaphore
        const auto& lastSignal = submit.signalSemaphores.back();
        m_submittedResources->latch({.semaphore=lastSignal.semaphore,.value=lastSignal.value},DeferredSubmitResourceDrop(submit));
        // Mark cmdbufs as done (wrongly but conservatively wrong)
        // We can't use `m_submittedResources` to mark them done, because the functor may run "late" in the future, after the cmdbuf has already been validly reset or resubmitted
        for (const auto& commandBuffer : submit.commandBuffers)
            commandBuffer.cmdbuf->m_state = commandBuffer.cmdbuf->getRecordingFlags().hasFlags(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT) ? IGPUCommandBuffer::STATE::INVALID:IGPUCommandBuffer::STATE::EXECUTABLE;
    }
    return result;
}

auto IQueue::waitIdle() -> RESULT
{
    const auto retval = waitIdle_impl();
    // everything will be murdered because every submitted semaphore so far will signal
    m_submittedResources->abortAll();
    return retval;
}

uint32_t IQueue::cullResources(const ISemaphore* sema)
{
    if (sema)
    {
        const auto& timelines = m_submittedResources->getTimelines();
        auto found = timelines.find(sema);
        if (found==timelines.end())
            return 0;
        return found->handler->poll().eventsLeft;
    }
    return m_submittedResources->poll().eventsLeft;
}

}