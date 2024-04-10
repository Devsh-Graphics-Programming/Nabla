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
		class DeferredSubmitResourceDrop final
		{
                using smart_ptr = core::smart_refctd_ptr<IBackendObject>;
				core::smart_refctd_dynamic_array<smart_ptr> m_resources;

			public:
				inline DeferredSubmitResourceDrop(const SSubmitInfo& info)
				{
                    // We could actually not hold any signal semaphore because you're expected to use the signal result somewhere else.
                    // However it's possible to you might only wait on one from the set and then drop the rest (UB) 
                    m_resources = core::make_refctd_dynamic_array<decltype(m_resources)>(info.signalSemaphores.size()-1+info.commandBuffers.size()+info.waitSemaphores.size());
                    auto outRes = m_resources->data();
                    for (const auto& sema : info.waitSemaphores)
                        *(outRes++) = smart_ptr(sema.semaphore);
                    for (const auto& cb : info.commandBuffers)
                        *(outRes++) = smart_ptr(cb.cmdbuf);
                    // We don't hold the last signal semaphore, because the timeline does as an Event trigger.
                    for (auto i=0u; i<info.signalSemaphores.size()-1; i++)
                        *(outRes++) = smart_ptr(info.signalSemaphores[i].semaphore);
				}
                DeferredSubmitResourceDrop(const DeferredSubmitResourceDrop& other) = delete;
				inline DeferredSubmitResourceDrop(DeferredSubmitResourceDrop&& other) : m_resources(nullptr)
				{
					this->operator=(std::move(other));
				}

                DeferredSubmitResourceDrop& operator=(const DeferredSubmitResourceDrop& other) = delete;
				inline DeferredSubmitResourceDrop& operator=(DeferredSubmitResourceDrop&& other)
				{
                    m_resources = std::move(other.m_resources);
                    other.m_resources = nullptr;
					return *this;
				}

                // always exhaustive poll, because we need to get rid of resources ASAP
                inline void operator()()
                {
                    m_resources = nullptr;
                }
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
        std::unique_ptr<MultiTimelineEventHandlerST<DeferredSubmitResourceDrop,false>> m_submittedResources;
        const ILogicalDevice* m_originDevice;
        const uint32_t m_familyIndex;
        const float m_priority;
        const core::bitflag<CREATE_FLAGS> m_flags;
};

NBL_ENUM_ADD_BITWISE_OPERATORS(IQueue::FAMILY_FLAGS)

}

#endif