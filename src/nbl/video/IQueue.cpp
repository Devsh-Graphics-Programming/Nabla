#include "nbl/video/IQueue.h"
#include "nbl/video/IGPUCommandBuffer.h"
#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/TimelineEventHandlers.h"

#include "git_info.h"
#define NBL_LOG_FUNCTION logger->log
#include "nbl/logging_macros.h"

namespace nbl::video
{

IQueue::IQueue(ILogicalDevice* originDevice, const uint32_t _famIx, const core::bitflag<CREATE_FLAGS> _flags, const float _priority)
    : m_originDevice(originDevice), m_familyIndex(_famIx), m_priority(_priority), m_flags(_flags)
{
    m_submittedResources = std::make_unique<MultiTimelineEventHandlerST<DeferredSubmitCallback,false>>(originDevice);
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
                    NBL_LOG_ERROR("Why on earth are you trying to submit a nullptr semaphore or to a wrong device!?");
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
                NBL_LOG_ERROR("Why on earth are you trying to submit a nullptr command buffer or to a wrong device!?");
                return RESULT::OTHER_ERROR;
            }

            const char* commandBufferDebugName = cmdbuf->getDebugName();
            if (!commandBufferDebugName)
                commandBufferDebugName = "";

            if (cmdbuf->getLevel()!=IGPUCommandPool::BUFFER_LEVEL::PRIMARY)
            {
                NBL_LOG_ERROR("Command buffer (%s, %p) is NOT PRIMARY LEVEL", commandBufferDebugName, cmdbuf);
                return RESULT::OTHER_ERROR;
            }
            if (cmdbuf->getState()!=IGPUCommandBuffer::STATE::EXECUTABLE)
            {
                NBL_LOG_ERROR("Command buffer (%s, %p) is NOT IN THE EXECUTABLE STATE", commandBufferDebugName, cmdbuf);
                return RESULT::OTHER_ERROR;
            }

            const auto& descriptorSetsRecord = cmdbuf->getBoundDescriptorSetsRecord();
            for (const auto& dsRecord : descriptorSetsRecord)
            {
                const auto& [ds, cachedDSVersion] = dsRecord;
                if (ds->getVersion() > cachedDSVersion)
                {
                    NBL_LOG(system::ILogger::ELL_WARNING, "Descriptor set(s) updated after being bound without UPDATE_AFTER_BIND. Invalidating command buffer (%s, %p)..", commandBufferDebugName, cmdbuf)
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


    if (result != RESULT::SUCCESS)
    {

        if (result == RESULT::DEVICE_LOST)
        {
            NBL_LOG_ERROR("Device lost");
            _NBL_DEBUG_BREAK_IF(true);
        }
        else
        {
            NBL_LOG_ERROR("Failed submit command buffers to the queue");
        }
        return result;
    }
    // poll for whatever is done, free up memory ASAP
    // we poll everything (full GC run) because we would release memory very slowly otherwise
    m_submittedResources->poll();
    //
    for (const auto& submit : _submits)
    {
        // hold onto the semaphores and commandbuffers until the submit signals the last semaphore
        const auto& lastSignal = submit.signalSemaphores.back();
        m_submittedResources->latch({.semaphore=lastSignal.semaphore,.value=lastSignal.value},DeferredSubmitCallback(submit));
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

IQueue::DeferredSubmitCallback::DeferredSubmitCallback(const SSubmitInfo& info)
{
    // We could actually not hold any signal semaphore because you're expected to use the signal result somewhere else.
    // However it's possible to you might only wait on one from the set and then drop the rest (UB) 
    m_resources = core::make_refctd_dynamic_array<decltype(m_resources)>(info.signalSemaphores.size()-1+info.commandBuffers.size()+info.waitSemaphores.size());
    auto outRes = m_resources->data();
    for (const auto& sema : info.waitSemaphores)
        *(outRes++) = smart_ptr(sema.semaphore);
    for (const auto& cb : info.commandBuffers)
    {
        *(outRes++) = smart_ptr(cb.cmdbuf);
        // get the TLAS BLAS tracking info and assign a pending build version number
        for (const auto& refSet : cb.cmdbuf->m_TLASToBLASReferenceSets)
        {
            const auto tlas = refSet.first;
            using iterator = decltype(refSet.second)::iterator;
            struct CustomIterator
            {
                inline bool operator!=(const CustomIterator& other) const {return ptr!=other.ptr;}

                inline CustomIterator operator++() {return {ptr++};}

                inline const IGPUBottomLevelAccelerationStructure* operator*() const {return dynamic_cast<const IGPUBottomLevelAccelerationStructure*>(ptr->get());}

                iterator ptr;
            };
            const auto buildVer = tlas->pushTrackedBLASes<CustomIterator>({refSet.second.begin()},{refSet.second.end()});
            // in theory could assert no duplicate entries, but thats obvious
            m_TLASBuilds[tlas] = buildVer;
        }
    }
    // We don't hold the last signal semaphore, because the timeline does as an Event trigger.
    for (auto i=0u; i<info.signalSemaphores.size()-1; i++)
        *(outRes++) = smart_ptr(info.signalSemaphores[i].semaphore);
    // copy the function object for the callback
    if (info.completionCallback)
        m_callback = *info.completionCallback;
}

IQueue::DeferredSubmitCallback& IQueue::DeferredSubmitCallback::operator=(DeferredSubmitCallback&& other)
{
    m_TLASBuilds = std::move(other.m_TLASBuilds);
    m_resources = std::move(other.m_resources);
    m_callback = std::move(other.m_callback);
    other.m_TLASBuilds.clear();
    other.m_resources = nullptr;
    other.m_callback = {};
	return *this;
}

// always exhaustive poll, because we need to get rid of resources ASAP
void IQueue::DeferredSubmitCallback::operator()()
{
    // all builds started before ours will now get overwritten (not exactly true, but without a better tracking system, this is the best we can do for now)
    for (const auto& build : m_TLASBuilds)
        build.first->clearTrackedBLASes(build.second);
    // then free all resources
    m_resources = nullptr;
    // then execute the callback
    if (m_callback)
    {
        m_callback();
        m_callback = {};
    }
}

} // namespace nbl::video

#include "nbl/undef_logging_macros.h"