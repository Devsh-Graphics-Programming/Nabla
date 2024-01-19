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

        // Vulkan: const VkSemaphore*
        virtual const void* getNativeHandle() const = 0;

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

        auto const& getCreationParams() const
        {
            return m_creationParams;
        }

    protected:
        ISemaphore(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params = {})
            : IBackendObject(std::move(dev))
            , m_creationParams(std::move(params))
        {}
        virtual ~ISemaphore() = default;

        const SCreationParams m_creationParams;
};

}
#endif