#ifndef _NBL_VIDEO_I_GPU_QUEUE_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_QUEUE_H_INCLUDED_

#include "nbl/video/IGPUFence.h"
#include "nbl/video/IGPUSemaphore.h"

namespace nbl::video
{

class IGPUCommandBuffer;

class IGPUQueue : public core::Interface, public core::Unmovable
{
    public:
        // when you don't want an ownership transfer
        constexpr static inline uint32_t FamilyIgnored = 0x7fffffffu;
        // for queues on the same device group, driver version, as indicated by deviceUUID and driverUUID
        constexpr static inline uint32_t FamilyExternal = 0x7ffffffeu;
        // any queue external, regardless of device or driver version
        constexpr static inline uint32_t FamilyForeign = 0x7ffffffdu;

        enum class FAMILY_FLAGS : uint8_t
        {
            NONE = 0,
            GRAPHICS_BIT = 0x01,
            COMPUTE_BIT = 0x02,
            TRANSFER_BIT = 0x04,
            SPARSE_BINDING_BIT = 0x08,
            PROTECTED_BIT = 0x10
        };
        enum class CREATE_FLAGS : uint32_t
        {
            NONE = 0x00u,
            PROTECTED_BIT = 0x01u
        };


        // for renderdoc and friends
        virtual bool startCapture() = 0;
        virtual bool endCapture() = 0;

        //
        struct SSubmitInfo
        {
            uint32_t waitSemaphoreCount = 0u;
            IGPUSemaphore*const * pWaitSemaphores = nullptr;
            const asset::PIPELINE_STAGE_FLAGS* pWaitDstStageMask = nullptr;
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
        virtual bool submit(const uint32_t _count, const SSubmitInfo* const _submits, IGPUFence* const _fence);

        // getters
        float getPriority() const { return m_priority; }
        uint32_t getFamilyIndex() const { return m_familyIndex; }
        CREATE_FLAGS getFlags() const { return m_flags; }

        constexpr static inline float DEFAULT_QUEUE_PRIORITY = 1.f;

        // OpenGL: const egl::CEGL::Context*
        // Vulkan: const VkQueue*
        virtual const void* getNativeHandle() const = 0;

    protected:
        //! `flags` takes bits from E_CREATE_FLAGS
        inline IGPUQueue(ILogicalDevice* originDevice, uint32_t _famIx, CREATE_FLAGS _flags, float _priority)
            : m_originDevice(originDevice), m_familyIndex(_famIx), m_priority(_priority), m_flags(_flags)
        {
        }

        virtual bool submit_impl(const uint32_t _count, const SSubmitInfo* _submits, IGPUFence* const _fence) = 0;

        inline bool markCommandBuffersAsPending(const uint32_t _count, const SSubmitInfo* _submits);
        bool markCommandBuffersAsDone(const uint32_t _count, const SSubmitInfo* _submits);

        const ILogicalDevice* m_originDevice;
        const uint32_t m_familyIndex;
        const float m_priority;
        const CREATE_FLAGS m_flags;
};

NBL_ENUM_ADD_BITWISE_OPERATORS(IGPUQueue::FAMILY_FLAGS)

}

#endif