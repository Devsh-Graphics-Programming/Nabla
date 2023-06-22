#ifndef _NBL_VIDEO_I_QUEUE_H_INCLUDED_
#define _NBL_VIDEO_I_QUEUE_H_INCLUDED_

#include "nbl/video/ISemaphore.h"

namespace nbl::video
{

class IGPUCommandBuffer;

class IQueue : public core::Interface, public core::Unmovable
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
            struct SSemaphoreInfo
            {
                ISemaphore* semaphore = nullptr;
                // depending on if you put it as the wait or signal semaphore, this is the value to wait for or to signal
                uint64_t value = 0u;
                core::bitflag<asset::PIPELINE_STAGE_FLAGS> stageMask = asset::PIPELINE_STAGE_FLAGS::NONE;
                //uint32_t deviceIndex = 0;
            };
            struct SCommandBufferInfo
            {
                IGPUCommandBuffer* cmdbuf = nullptr;
                //uint32_t deviceMask = 0b1u;
            };

            // TODO: flags/bitfields
            uint32_t waitSemaphoreCount = 0u;
            uint32_t commandBufferCount = 0u;
            uint32_t signalSemaphoreCount = 0u;
            const SSemaphoreInfo* pWaitSemaphores = nullptr;
            const SCommandBufferInfo* commandBuffers = nullptr;
            const SSemaphoreInfo* pSignalSemaphores = nullptr;

            inline bool valid() const
            {
                if (waitSemaphoreCount>0u && !pWaitSemaphores)
                    return false;
                if (commandBufferCount>0u && !commandBuffers)
                    return false;
                if (signalSemaphoreCount>0u && !pSignalSemaphores)
                    return false;
                // wait & work | work & signal | wait & signal
                return waitSemaphoreCount&&commandBufferCount || commandBufferCount&&signalSemaphoreCount || waitSemaphoreCount&&signalSemaphoreCount;
            }
        };
        virtual bool submit(const uint32_t _count, const SSubmitInfo* const _submits);

        // getters
        inline CREATE_FLAGS getFlags() const { return m_flags; }
        inline uint32_t getFamilyIndex() const { return m_familyIndex; }
        inline float getPriority() const { return m_priority; }

        constexpr static inline float DEFAULT_QUEUE_PRIORITY = 1.f;

        // Vulkan: const VkQueue*
        virtual const void* getNativeHandle() const = 0;

    protected:
        //! `flags` takes bits from E_CREATE_FLAGS
        inline IQueue(ILogicalDevice* originDevice, uint32_t _famIx, CREATE_FLAGS _flags, float _priority)
            : m_originDevice(originDevice), m_familyIndex(_famIx), m_priority(_priority), m_flags(_flags)
        {
        }

        virtual bool submit_impl(const uint32_t _count, const SSubmitInfo* const _submits) = 0;

        const ILogicalDevice* m_originDevice;
        const uint32_t m_familyIndex;
        const float m_priority;
        const CREATE_FLAGS m_flags;
};

NBL_ENUM_ADD_BITWISE_OPERATORS(IQueue::FAMILY_FLAGS)

}

#endif