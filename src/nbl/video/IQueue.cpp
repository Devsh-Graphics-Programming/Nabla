#include "nbl/video/IQueue.h"
#include "nbl/video/IGPUCommandBuffer.h"
#include "nbl/video/ILogicalDevice.h"

namespace nbl::video
{

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
    // mark cmdbufs as done (wrongly but conservatively wrong)
    for (const auto& submit : _submits)
    for (const auto& commandBuffer : submit.commandBuffers)
        commandBuffer.cmdbuf->m_state = commandBuffer.cmdbuf->getRecordingFlags().hasFlags(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT) ? IGPUCommandBuffer::STATE::INVALID:IGPUCommandBuffer::STATE::EXECUTABLE;
    return result;
}

}