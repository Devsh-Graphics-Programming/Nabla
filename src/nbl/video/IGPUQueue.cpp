#include "nbl/video/IGPUQueue.h"
#include "nbl/video/IGPUCommandBuffer.h"
#include "nbl/video/ILogicalDevice.h"

namespace nbl::video
{

bool IGPUQueue::submit(const uint32_t _count, const SSubmitInfo* _submits, IGPUFence* const _fence)
{
    if (_submits == nullptr)
        return false;

    for (uint32_t i = 0u; i < _count; ++i)
    {
        auto& submit = _submits[i];
        for (uint32_t j = 0u; j < submit.commandBufferCount; ++j)
        {
            if (submit.commandBuffers[j] == nullptr)
                return false;

            assert(submit.commandBuffers[j]->getLevel() == IGPUCommandBuffer::LEVEL::PRIMARY);
            assert(submit.commandBuffers[j]->getState() == IGPUCommandBuffer::STATE::EXECUTABLE);

            if (submit.commandBuffers[j]->getLevel() != IGPUCommandBuffer::LEVEL::PRIMARY)
                return false;
            if (submit.commandBuffers[j]->getState() != IGPUCommandBuffer::STATE::EXECUTABLE)
                return false;

            const auto& descriptorSetsRecord = submit.commandBuffers[j]->getBoundDescriptorSetsRecord();
            for (const auto& dsRecord : descriptorSetsRecord)
            {
                const auto& [ds, cachedDSVersion] = dsRecord;
                if (ds->getVersion() > cachedDSVersion)
                {
                    const char* commandBufferDebugName = submit.commandBuffers[j]->getDebugName();
                    if (commandBufferDebugName)
                        m_originDevice->getPhysicalDevice()->getDebugCallback()->getLogger()->log("Descriptor set(s) updated after being bound without UPDATE_AFTER_BIND. Invalidating command buffer (%s, %p)..", system::ILogger::ELL_ERROR, commandBufferDebugName, submit.commandBuffers[i]);
                    else
                        m_originDevice->getPhysicalDevice()->getDebugCallback()->getLogger()->log("Descriptor set(s) updated after being bound without UPDATE_AFTER_BIND. Invalidating command buffer (%p)..", system::ILogger::ELL_ERROR, submit.commandBuffers[i]);

                    submit.commandBuffers[j]->m_state = IGPUCommandBuffer::STATE::INVALID;
                    return false;
                }
            }
        }
    }
    return submit_impl(_count,_submits,_fence);
}

bool IGPUQueue::markCommandBuffersAsPending(const uint32_t _count, const SSubmitInfo* _submits)
{
    if(_submits == nullptr)
        return false;
    for (uint32_t i = 0u; i < _count; ++i)
    {
        auto& submit = _submits[i];
        for (uint32_t j = 0u; j < submit.commandBufferCount; ++j)
        {
            auto& cmdbuf = submit.commandBuffers[j];
            if(cmdbuf == nullptr)
                return false;
            submit.commandBuffers[j]->m_state = IGPUCommandBuffer::STATE::PENDING;
        }
    }
    return true;
}
    
bool IGPUQueue::markCommandBuffersAsDone(const uint32_t _count, const SSubmitInfo* _submits)
{
    if(_submits == nullptr)
        return false;
    for (uint32_t i = 0u; i < _count; ++i)
    {
        auto& submit = _submits[i];
        for (uint32_t j = 0u; j < submit.commandBufferCount; ++j)
        {
            auto& cmdbuf = submit.commandBuffers[j];
            if(cmdbuf == nullptr)
                return false;

            if (cmdbuf->isResettable())
                cmdbuf->m_state = IGPUCommandBuffer::STATE::EXECUTABLE;
            else
                cmdbuf->m_state = IGPUCommandBuffer::STATE::INVALID;
        }
    }
    return true;
}

}