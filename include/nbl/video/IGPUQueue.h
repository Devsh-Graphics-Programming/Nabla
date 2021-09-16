#ifndef __NBL_VIDEO_I_GPU_QUEUE_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_QUEUE_H_INCLUDED__

#include "nbl/video/decl/IBackendObject.h"
#include "nbl/video/IGPUCommandBuffer.h"

namespace nbl::video
{

class IGPUFence;
class IGPUSemaphore;
class ISwapchain;

class IGPUQueue : public core::Interface, public core::Unmovable
{
    public:
        enum E_CREATE_FLAGS : uint32_t
        {
            ECF_PROTECTED_BIT = 0x01
        };

        struct SSubmitInfo
        {
            uint32_t waitSemaphoreCount = 0u;
            IGPUSemaphore*const * pWaitSemaphores = nullptr;
            const asset::E_PIPELINE_STAGE_FLAGS* pWaitDstStageMask = nullptr;
            uint32_t signalSemaphoreCount = 0u;
            IGPUSemaphore*const * pSignalSemaphores = nullptr;
            uint32_t commandBufferCount = 0u;
            IGPUCommandBuffer*const * commandBuffers = nullptr;
        };
        struct SPresentInfo
        {
            uint32_t waitSemaphoreCount;
            IGPUSemaphore*const * waitSemaphores;
            uint32_t swapchainCount;
            ISwapchain*const * swapchains;
            const uint32_t* imgIndices;
        };

        //! `flags` takes bits from E_CREATE_FLAGS
        IGPUQueue(ILogicalDevice* originDevice, uint32_t _famIx, E_CREATE_FLAGS _flags, float _priority)
            : m_originDevice(originDevice), m_flags(_flags), m_familyIndex(_famIx), m_priority(_priority)
        {

        }

        // for renderdoc and friends
        virtual bool startCapture() {}// = 0;
        virtual bool endCapture() {}// = 0;

        //
        virtual bool submit(uint32_t _count, const SSubmitInfo* _submits, IGPUFence* _fence) = 0;

        //
        virtual bool present(const SPresentInfo& info) = 0;

        // getters
        float getPriority() const { return m_priority; }
        uint32_t getFamilyIndex() const { return m_familyIndex; }
        E_CREATE_FLAGS getFlags() const { return m_flags; }

    protected:
        const uint32_t m_familyIndex;
        const E_CREATE_FLAGS m_flags;
        const float m_priority;
        ILogicalDevice* m_originDevice;
};

inline bool IGPUQueue::submit(uint32_t _count, const SSubmitInfo* _submits, IGPUFence* _fence)
{
    for (uint32_t i = 0u; i < _count; ++i)
    {
        auto& submit = _submits[i];
        for (uint32_t j = 0u; j < submit.commandBufferCount; ++j)
        {
            assert(submit.commandBuffers[j]->getLevel() == IGPUCommandBuffer::EL_PRIMARY);
            assert(submit.commandBuffers[j]->getState() == IGPUCommandBuffer::ES_EXECUTABLE);

            if (submit.commandBuffers[j]->getLevel() != IGPUCommandBuffer::EL_PRIMARY)
                return false;
            if (submit.commandBuffers[j]->getState() != IGPUCommandBuffer::ES_EXECUTABLE)
                return false;
        }
    }
    return true;
}

}

#endif