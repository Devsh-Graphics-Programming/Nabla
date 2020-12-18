#ifndef __IRR_I_GPU_QUEUE_H_INCLUDED__
#define __IRR_I_GPU_QUEUE_H_INCLUDED__

#include <nbl/core/IReferenceCounted.h>
#include <nbl/video/IGPUPrimaryCommandBuffer.h>
#include <nbl/video/IGPUSemaphore.h>
#include <nbl/video/IGPUFence.h>

namespace nbl {
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
        uint32_t waitSemaphoreCount;
        IGPUSemaphore** pWaitSemaphores;
        const E_PIPELINE_STAGE_FLAGS* pWaitDstStageMask;
        uint32_t signalSemaphoreCount;
        IGPUSemaphore** pSignalSemaphores;
        uint32_t commandBufferCount;
        IGPUPrimaryCommandBuffer** commandBuffers;
    };

    //! `flags` takes bits from E_CREATE_FLAGS
    IGPUQueue(uint32_t _famIx, uint32_t _flags, float _priority)
        : m_flags(_flags), m_familyIndex(_famIx), m_priority(_priority)
    {

    }

    virtual void submit(uint32_t _count, const SSubmitInfo* _submits, IGPUFence* _fence) = 0;

protected:
    const uint32_t m_familyIndex;
    //takes bits from E_CREATE_FLAGS
    const uint32_t m_flags;
    const float m_priority;
};

}}

#endif