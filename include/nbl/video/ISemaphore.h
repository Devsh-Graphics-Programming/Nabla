#ifndef _NBL_VIDEO_I_SEMAPHORE_H_INCLUDED_
#define _NBL_VIDEO_I_SEMAPHORE_H_INCLUDED_


#include "nbl/core/IReferenceCounted.h"

#include <chrono>

#include "nbl/video/decl/IBackendObject.h"


namespace nbl::video
{

class ISemaphore : public IBackendObject
{
    public:

        //! Flags for imported/exported allocation
        enum E_EXTERNAL_HANDLE_TYPE : uint32_t
        {
            EHT_NONE = 0x00000000,
            EHT_OPAQUE_FD = 0x00000001,
            EHT_OPAQUE_WIN32 = 0x00000002,
            EHT_OPAQUE_WIN32_KMT = 0x00000004,
            EHT_D3D12_FENCE = 0x00000008,
            EHT_SYNC_FD = 0x00000010,
        };

        //!
        struct SCreationParams
        {
            // A Pre-Destroy-Step is called out just before a `vkDestory` or `glDelete`, this is only useful for "imported" resources
            std::unique_ptr<ICleanup> preDestroyCleanup = nullptr;
            // A Post-Destroy-Step is called in this class' destructor, this is only useful for "imported" resources
            std::unique_ptr<ICleanup> postDestroyCleanup = nullptr;
            // Thus the destructor will skip the call to `vkDestroy` or `glDelete` on the handle, this is only useful for "imported" objects
            bool skipHandleDestroy = false;
            // Handle Type for external resources
            core::bitflag<E_EXTERNAL_HANDLE_TYPE> externalHandleTypes = EHT_NONE;
            //! Imports the given handle  if externalHandle != nullptr && externalMemoryHandleType != EHT_NONE
            //! Creates exportable memory if externalHandle == nullptr && externalMemoryHandleType != EHT_NONE
            ExternalHandleType externalHandle = nullptr;
        };

        // basically a pool function
        virtual uint64_t getCounterValue() const = 0;

        //! Basically the counter can only monotonically increase with time (ergo the "timeline"):
        // 1. `value` must have a value greater than the current value of the semaphore (what you'd get from `getCounterValue()`)
        // 2. `value` must be less than the value of any pending semaphore signal operations (this is actually more complicated)
        // Current pending signal operations can complete in any order, unless there's an execution dependency between them,
        // this will change the current value of the semaphore. Consider a semaphore with current value of 2 and pending signals of 3,4,5;
        // without any execution dependencies, you can only signal a value higher than 2 but less than 3 which is impossible.
        virtual void signal(const uint64_t value) = 0;

        // We don't provide waits as part of the semaphore (cause you can await multiple at once with ILogicalDevice),
        // but don't want to pollute ILogicalDevice with lots of enums and structs
        struct SWaitInfo
        {
            const ISemaphore* semaphore = nullptr;
            uint64_t value = 0;
        };
        enum class WAIT_RESULT : uint8_t
        {
            TIMEOUT,
            SUCCESS,
            DEVICE_LOST,
            _ERROR
        };

        // Utility structs
        class future_base_t
        {
            public:
                inline future_base_t(future_base_t&&) = default;
                inline future_base_t& operator=(future_base_t&&) = default;

                inline bool blocking() const
                {
                    return m_semaphore.get();
                }

                inline bool ready() const
                {
                    if (m_semaphore)
                        return m_semaphore->getCounterValue()>=m_waitValue;
                    return true;
                }

                inline operator SWaitInfo() const {return {m_semaphore.get(),m_waitValue};}

                NBL_API2 WAIT_RESULT wait() const;

            protected:
                // the base class is not directly usable
                inline future_base_t() = default;
                // derived won't be copyable
                future_base_t(const future_base_t&) = delete;
                future_base_t& operator=(const future_base_t&) = delete;

                // smartpointer cause lifetime needs to be maintained
                core::smart_refctd_ptr<const ISemaphore> m_semaphore;
                uint64_t m_waitValue;
        };
        // Similar to `ISystem::future_t` but NOT thread-safe and the condition we wait on is signalled by the Device
        template<typename T>
        class future_t : private core::StorageTrivializer<T>, public future_base_t
        {
                using storage_t = core::StorageTrivializer<T>;
                using this_t = future_t<T>;

            public:
                inline future_t(this_t&& other) noexcept : future_base_t(std::move(static_cast<future_base_t&>(other)))
                {
                    if constexpr (!std::is_void_v<T>)
                        storage_t::construct(std::move(*other.getStorage()));
                }
                template<typename... Args>
                inline future_t(Args&&... args) noexcept
                {
                    storage_t::construct(std::forward<Args>(args)...);
                }
                inline ~future_t()
                {
                    const auto success = wait();
                    assert(success!=WAIT_RESULT::TIMEOUT);
                    storage_t::destruct();
                }

                inline this_t& operator=(this_t&& rhs)
                {
                    future_base_t::operator=(std::move<future_base_t>(rhs));
                    if constexpr (!std::is_void_v<T>)
                        *storage_t::getStorage() = std::move(*rhs.getStorage());
                    return *this;
                }

                inline void set(const SWaitInfo& wait)
                {
                    m_semaphore = core::smart_refctd_ptr<const ISemaphore>(wait.semaphore);
                    m_waitValue = wait.value;
                }
                template<std::copyable U=T> requires std::is_same_v<T,U>
                inline void set(U&& val)
                {
                    *storage_t::getStorage() = std::move(val);
                }

                inline const T* get() const
                {
                    if (ready())
                        return storage_t::getStorage();
                    return nullptr;
                }

                template<std::copyable U=T> requires std::is_same_v<T,U>
                inline U copy() const
                {
                    const auto success = wait();
                    assert(success!=WAIT_RESULT::TIMEOUT);
                    return *get();
                }

                template<std::movable U=T> requires std::is_same_v<T,U>
                inline void move_into(U& dst)
                {
                    const auto success = wait();
                    assert(success!=WAIT_RESULT::TIMEOUT);
                    dst = std::move(*get());
                }
        };

        // Vulkan: const VkSemaphore*
        virtual const void* getNativeHandle() const = 0;

        const SCreationParams& getCreationParams() const { return m_creationParams; }

    protected:
        inline ISemaphore(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& creationParams) : IBackendObject(std::move(dev)), m_creationParams(std::move(creationParams)) {}
        virtual ~ISemaphore() = default;

        SCreationParams m_creationParams;
};

}
#endif