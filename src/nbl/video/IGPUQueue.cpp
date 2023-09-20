#include "nbl/video/IGPUQueue.h"
#include "nbl/video/ILogicalDevice.h"

namespace nbl::video
{
bool IGPUQueue::submit(uint32_t _count, const SSubmitInfo* _submits, IGPUFence* _fence)
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

            assert(submit.commandBuffers[j]->getLevel() == IGPUCommandBuffer::EL_PRIMARY);
            assert(submit.commandBuffers[j]->getState() == IGPUCommandBuffer::ES_EXECUTABLE);

            if (submit.commandBuffers[j]->getLevel() != IGPUCommandBuffer::EL_PRIMARY)
                return false;
            if (submit.commandBuffers[j]->getState() != IGPUCommandBuffer::ES_EXECUTABLE)
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

                    submit.commandBuffers[j]->setState(IGPUCommandBuffer::ES_INVALID);
                    return false;
                }
            }
        }
    }
    return true;
}
}