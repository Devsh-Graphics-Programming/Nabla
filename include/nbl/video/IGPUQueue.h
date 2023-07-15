#ifndef __NBL_VIDEO_I_GPU_QUEUE_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_QUEUE_H_INCLUDED__

#include "nbl/video/decl/IBackendObject.h"
#include "nbl/video/IGPUCommandBuffer.h"

namespace nbl::video
{

class IGPUFence;
class IGPUSemaphore;

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

            inline bool isValid() const
            {
                if (waitSemaphoreCount > 0u && (pWaitSemaphores == nullptr || pWaitDstStageMask == nullptr))
                    return false;
                if (signalSemaphoreCount > 0u && pSignalSemaphores == nullptr)
                    return false;
                if (commandBufferCount > 0u && commandBuffers == nullptr)
                    return false;
                return true;
            }
        };

        //! `flags` takes bits from E_CREATE_FLAGS
        IGPUQueue(ILogicalDevice* originDevice, uint32_t _famIx, E_CREATE_FLAGS _flags, float _priority)
            : m_originDevice(originDevice), m_flags(_flags), m_familyIndex(_famIx), m_priority(_priority)
        {

        }

        // for renderdoc and friends
        virtual bool startCapture() = 0;
        virtual bool endCapture() = 0;

        virtual bool insertDebugMarker(const char* name, const core::vector4df_SIMD& color) = 0;
        virtual bool beginDebugMarker(const char* name, const core::vector4df_SIMD& color) = 0;
        virtual bool endDebugMarker() = 0;

        //
        virtual bool submit(uint32_t _count, const SSubmitInfo* _submits, IGPUFence* _fence) = 0;

        // getters
        float getPriority() const { return m_priority; }
        uint32_t getFamilyIndex() const { return m_familyIndex; }
        E_CREATE_FLAGS getFlags() const { return m_flags; }

        inline constexpr static float DEFAULT_QUEUE_PRIORITY = 1.f;

        // OpenGL: const egl::CEGL::Context*
        // Vulkan: const VkQueue*
        virtual const void* getNativeHandle() const = 0;

    protected:
        inline bool markCommandBuffersAsPending(uint32_t _count, const SSubmitInfo* _submits)
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
                    submit.commandBuffers[j]->setState(IGPUCommandBuffer::ES_PENDING);
                }
            }
            return true;
        }
    
        inline bool markCommandBuffersAsDone(uint32_t _count, const SSubmitInfo* _submits)
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
                        cmdbuf->setState(IGPUCommandBuffer::ES_EXECUTABLE);
                    else
                        cmdbuf->setState(IGPUCommandBuffer::ES_INVALID);
                }
            }
            return true;
        }

        const uint32_t m_familyIndex;
        const E_CREATE_FLAGS m_flags;
        const float m_priority;
        ILogicalDevice* m_originDevice;
};

}

#endif