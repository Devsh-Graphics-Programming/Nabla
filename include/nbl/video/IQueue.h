#ifndef _NBL_VIDEO_I_QUEUE_H_INCLUDED_
#define _NBL_VIDEO_I_QUEUE_H_INCLUDED_

#include "nbl/video/ISemaphore.h"

namespace nbl::video
{

class IGPUCommandBuffer;

template<typename Functor,bool RefcountTheDevice>
class MultiTimelineEventHandlerST;

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
#if 0
        // When dealing with external/foreign queues treat `other` as nullptr
        inline bool needsOwnershipTransfer(const IQueue* other) const // TODO: move into IDeviceMemoryBacked
        {
            if (!other)
                return true;

            if (m_familyIndex==other->m_familyIndex)
                return false;

            // TODO: take into account concurrent sharing indices, but then we'll need to remember the concurrent sharing family indices
            return true;
        }
#endif

        // for renderdoc and friends
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
        // We actually make our Queue Abstraction keep track of Commandbuffers and Semaphores used in a submit until its done
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
            // No guarantees are given about when it will execute, except that it will execute:
            // 1) after the `signalSemaphore.back()` signals
            // 2) in order w.r.t. all other submits on this queue
            // 3) after all lifetime tracking has been performed (so transient resources will already be dead!)
            // NOTE: This `std::function` WILL be copied! 
            std::function<void()>* completionCallback = nullptr;

            inline bool valid() const
            {
                // need at least one semaphore to keep the submit resources alive
                if (signalSemaphores.empty())
                    return false;
                return true;
            }
        };
        NBL_API2 virtual RESULT submit(const std::span<const SSubmitInfo> _submits);
        // Wait idle also makes sure all past submits drop their resources
        NBL_API2 virtual RESULT waitIdle();
        // You can call this to force an early check on whether past submits have completed and hasten when the refcount gets dropped.
        // Normally its the next call to `submit` that polls the event timeline for completions.
        // If you want to only check a particular semaphore's timeline, then set `sema` to non nullptr.
        NBL_API2 virtual uint32_t cullResources(const ISemaphore* sema=nullptr);

        // we cannot derive from IBackendObject because we can't derive from IReferenceCounted
        inline const ILogicalDevice* getOriginDevice() const {return m_originDevice;}
        inline bool wasCreatedBy(const ILogicalDevice* device) const {return device==m_originDevice;}
        // Vulkan: const VkQueue*
        virtual const void* getNativeHandle() const = 0;
        
		// only public because MultiTimelineEventHandlerST needs to know about it
		class DeferredSubmitCallback final
		{
                //
                struct STLASBuildMetadata
                {
                    core::unordered_set<IGPUTopLevelAccelerationStructure::blas_smart_ptr_t> m_BLASes;
                    uint32_t m_buildVer;
                };
                core::unordered_map<IGPUTopLevelAccelerationStructure*,STLASBuildMetadata> m_TLASToBLASReferenceSets;
                //
                using smart_ptr = core::smart_refctd_ptr<IBackendObject>;
                core::smart_refctd_dynamic_array<smart_ptr> m_resources;
                //
                std::function<void()> m_callback;

			public:
                DeferredSubmitCallback(const SSubmitInfo& info);
                DeferredSubmitCallback(const DeferredSubmitCallback& other) = delete;
				inline DeferredSubmitCallback(DeferredSubmitCallback&& other) : m_resources(nullptr)
				{
					this->operator=(std::move(other));
				}

                DeferredSubmitCallback& operator=(const DeferredSubmitCallback& other) = delete;
                DeferredSubmitCallback& operator=(DeferredSubmitCallback&& other);

                // always exhaustive poll, because we need to get rid of resources ASAP
                void operator()();
		};

    protected:
        NBL_API2 IQueue(ILogicalDevice* originDevice, const uint32_t _famIx, const core::bitflag<CREATE_FLAGS> _flags, const float _priority);
        // As IQueue is logically an integral part of the `ILogicalDevice` the destructor will only run during `~ILogicalDevice` which means `waitIdle` already been called
        inline ~IQueue()
        {
            //while (cullResources()) {} // deleter of `m_submittedResources` calls dtor which will do this
        }

        friend class CThreadSafeQueueAdapter;
        virtual RESULT submit_impl(const std::span<const SSubmitInfo> _submits) = 0;
        virtual RESULT waitIdle_impl() const = 0;

        // Refcounts all resources used by Pending Submits, gets occasionally cleared out
        std::unique_ptr<MultiTimelineEventHandlerST<DeferredSubmitCallback,false>> m_submittedResources;
        const ILogicalDevice* m_originDevice;
        const uint32_t m_familyIndex;
        const float m_priority;
        const core::bitflag<CREATE_FLAGS> m_flags;
};

NBL_ENUM_ADD_BITWISE_OPERATORS(IQueue::FAMILY_FLAGS)

}

#endif