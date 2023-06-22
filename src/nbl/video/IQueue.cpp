#include "nbl/video/IQueue.h"
#include "nbl/video/IGPUCommandBuffer.h"
#include "nbl/video/ILogicalDevice.h"

namespace nbl::video
{

bool IQueue::submit(const uint32_t _count, const SSubmitInfo* const _submits)
{
    if (!_submits || _count==0u)
        return false;

    auto* logger = m_originDevice->getPhysicalDevice()->getDebugCallback()->getLogger();
    for (uint32_t i=0u; i<_count; ++i)
    {
        auto& submit = _submits[i];
        if (!submit.valid())
            return false;

        auto invalidSemaphores = [this,logger](const uint32_t count, const SSubmitInfo::SSemaphoreInfo* semaphoreInfo) -> bool
        {
            for (uint32_t j=0u; j<count; ++j)
            {
                auto* sema = semaphoreInfo[j].semaphore;
                if (!sema || !sema->wasCreatedBy(m_originDevice))
                {
                    logger->log("Why on earth are you trying to submit a nullptr command buffer or to a wrong device!?", system::ILogger::ELL_ERROR);
                    return true;
                }
            }
            return false;
        };
        if (invalidSemaphores(submit.waitSemaphoreCount,submit.pWaitSemaphores) || invalidSemaphores(submit.signalSemaphoreCount,submit.pSignalSemaphores))
            return false;

        for (uint32_t j=0u; j<submit.commandBufferCount; ++j)
        {
            auto* cmdbuf = submit.commandBuffers[j].cmdbuf;
            if (!cmdbuf || !cmdbuf->wasCreatedBy(m_originDevice))
            {
                logger->log("Why on earth are you trying to submit a nullptr command buffer or to a wrong device!?", system::ILogger::ELL_ERROR);
                return false;
            }

            const char* commandBufferDebugName = cmdbuf->getDebugName();
            if (!commandBufferDebugName)
                commandBufferDebugName = "";

            if (cmdbuf->getLevel()!=IGPUCommandBuffer::LEVEL::PRIMARY)
            {
                logger->log("Command buffer (%s, %p) is NOT PRIMARY LEVEL", system::ILogger::ELL_ERROR, commandBufferDebugName, cmdbuf);
                return false;
            }
            if (cmdbuf->getState()!=IGPUCommandBuffer::STATE::EXECUTABLE)
            {
                logger->log("Command buffer (%s, %p) is NOT IN THE EXECUTABLE STATE", system::ILogger::ELL_ERROR, commandBufferDebugName, cmdbuf);
                return false;
            }

            const auto& descriptorSetsRecord = cmdbuf->getBoundDescriptorSetsRecord();
            for (const auto& dsRecord : descriptorSetsRecord)
            {
                const auto& [ds, cachedDSVersion] = dsRecord;
                if (ds->getVersion() > cachedDSVersion)
                {
                    logger->log("Descriptor set(s) updated after being bound without UPDATE_AFTER_BIND. Invalidating command buffer (%s, %p)..", system::ILogger::ELL_WARNING, commandBufferDebugName, cmdbuf);
                    cmdbuf->m_state = IGPUCommandBuffer::STATE::INVALID;
                    return false;
                }
            }
        }
    }

    // mark cmdbufs as pending
    for (uint32_t i=0u; i<_count; ++i)
    for (uint32_t j=0u; j<_submits[i].commandBufferCount; ++j)
        _submits[i].commandBuffers[j].cmdbuf->m_state = IGPUCommandBuffer::STATE::PENDING;
    // do the submit
    if (!submit_impl(_count,_submits))
        return false;
    // mark cmdbufs as done
    for (uint32_t i=0u; i<_count; ++i)
    for (uint32_t j=0u; j<_submits[i].commandBufferCount; ++j)
    {
        auto* cmdbuf = _submits[i].commandBuffers[j].cmdbuf;
        cmdbuf->m_state = cmdbuf->isResettable() ? IGPUCommandBuffer::STATE::EXECUTABLE:IGPUCommandBuffer::STATE::INVALID;
    }
    return true;
}

}