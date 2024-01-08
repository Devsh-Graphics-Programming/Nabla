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

        //
        constexpr static inline float DEFAULT_QUEUE_PRIORITY = 1.f;

        //
        enum class FAMILY_FLAGS : uint8_t
        {
            NONE = 0,
            GRAPHICS_BIT = 0x01,
            COMPUTE_BIT = 0x02,
            TRANSFER_BIT = 0x04,
            SPARSE_BINDING_BIT = 0x08,
            PROTECTED_BIT = 0x10
        };
        enum class CREATE_FLAGS : uint8_t
        {
            NONE = 0x00u,
            PROTECTED_BIT = 0x01u
        };

        // getters
        inline core::bitflag<CREATE_FLAGS> getFlags() const { return m_flags; }
        inline uint32_t getFamilyIndex() const { return m_familyIndex; }
        inline float getPriority() const { return m_priority; }

        // When dealing with external/foreign queues treat `other` as nullptr
        inline bool needsOwnershipTransfer(const IQueue* other) const
        {
            if (!other)
                return true;

            if (m_familyIndex==other->m_familyIndex)
                return false;

            // TODO: take into account concurrent sharing indices, but then we'll need to remember the concurrent sharing family indices
            return true;
        }


        // for renderdoc and friends
        virtual bool startCapture() = 0;
        virtual bool endCapture() = 0;
        virtual bool insertDebugMarker(const char* name, const core::vector4df_SIMD& color = core::vector4df_SIMD(1.0, 1.0, 1.0, 1.0)) = 0;
        virtual bool beginDebugMarker(const char* name, const core::vector4df_SIMD& color = core::vector4df_SIMD(1.0, 1.0, 1.0, 1.0)) = 0;
        virtual bool endDebugMarker() = 0;

        //
        enum class RESULT : uint8_t
        {
            SUCCESS,
            DEVICE_LOST,
            OTHER_ERROR
        };
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
            std::span<const SSemaphoreInfo> waitSemaphores = {};
            std::span<const SCommandBufferInfo> commandBuffers = {};
            std::span<const SSemaphoreInfo> signalSemaphores = {};

            inline bool valid() const
            {
                // any two being empty is wrong
                if (commandBuffers.empty() && signalSemaphores.empty()) // wait and do nothing
                    return false;
                if (waitSemaphores.empty() && signalSemaphores.empty()) // work without sync
                    return false;
                if (waitSemaphores.empty() && commandBuffers.empty()) // signal without doing work first
                    return false;
                return true;
            }
        };
        virtual RESULT submit(const std::span<const SSubmitInfo> _submits);
        //
        virtual RESULT waitIdle() const = 0;

        // we cannot derive from IBackendObject because we can't derive from IReferenceCounted
        inline bool wasCreatedBy(const ILogicalDevice* device) const { return device == m_originDevice; }
        // Vulkan: const VkQueue*
        virtual const void* getNativeHandle() const = 0;

    protected:
        //! `flags` takes bits from E_CREATE_FLAGS
        inline IQueue(const ILogicalDevice* originDevice, const uint32_t _famIx, const core::bitflag<CREATE_FLAGS> _flags, const float _priority)
            : m_originDevice(originDevice), m_familyIndex(_famIx), m_priority(_priority), m_flags(_flags) {}

        friend class CThreadSafeQueueAdapter;
        virtual RESULT submit_impl(const std::span<const SSubmitInfo> _submits) = 0;

        const ILogicalDevice* m_originDevice;
        const uint32_t m_familyIndex;
        const float m_priority;
        const core::bitflag<CREATE_FLAGS> m_flags;
};

NBL_ENUM_ADD_BITWISE_OPERATORS(IQueue::FAMILY_FLAGS)

}

#endif