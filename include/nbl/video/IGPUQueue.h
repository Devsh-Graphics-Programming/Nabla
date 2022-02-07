#ifndef __IRR_I_GPU_QUEUE_H_INCLUDED__
#define __IRR_I_GPU_QUEUE_H_INCLUDED__

#include <nbl/core/IReferenceCounted.h>
#include <nbl/video/IGPUPrimaryCommandBuffer.h>
#include "IDriverFence.h"

namespace nbl
{
namespace video
{
class IGPUQueue : public core::IReferenceCounted
{
public:
    enum E_CREATE_FLAGS : uint32_t
    {
        ECF_PROTECTED_BIT = 0x01
    };

    struct SSubmitInfo
    {
        //uint32_t waitSemaphoreCount;
        //const VkSemaphore* pWaitSemaphores;
        //const VkPipelineStageFlags* pWaitDstStageMask;
        //uint32_t signalSemaphoreCount;
        //const VkSemaphore* pSignalSemaphores;
        uint32_t commandBufferCount;
        const IGPUPrimaryCommandBuffer** commandBuffers;
    };

    //! `flags` takes bits from E_CREATE_FLAGS
    IGPUQueue(uint32_t _famIx, uint32_t _flags, float _priority)
        : m_flags(_flags), m_familyIndex(_famIx), m_priority(_priority)
    {
    }

    virtual void submit(uint32_t _count, const SSubmitInfo* _submits, IDriverFence* _fence)
    {
        for(uint32_t i = 0u; i < _count; ++i)
            submit(_submits[i]);
    }

protected:
    void submit(const IGPUPrimaryCommandBuffer* _cmdbuf)
    {
        auto* cmdbuf = const_cast<IGPUPrimaryCommandBuffer*>(_cmdbuf);
        cmdbuf->setState(IGPUCommandBuffer::ES_PENDING);
        /*Once execution of all submissions of a command buffer complete, it moves from the pending state, back to the executable state.
        If a command buffer was recorded with the VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT flag, it instead moves to the invalid state.
        */
    }
    void submit(const SSubmitInfo& _submit)
    {
        for(uint32_t i = 0u; i < _submit.commandBufferCount; ++i)
        {
            const auto* cmdbuf = _submit.commandBuffers[i];
            submit(cmdbuf);
        }
    }
    void submit_epilogue(IDriverFence* _fence)
    {
        if(_fence)
            _fence->waitCPU(9999999999ull);
    }

    const uint32_t m_familyIndex;
    //takes bits from E_CREATE_FLAGS
    const uint32_t m_flags;
    const float m_priority;
};

}
}

#endif