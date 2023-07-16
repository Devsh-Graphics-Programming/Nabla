#ifndef __NBL_I_GPU_SEMAPHORE_H_INCLUDED__
#define __NBL_I_GPU_SEMAPHORE_H_INCLUDED__


#include "nbl/core/IReferenceCounted.h"

#include "nbl/video/decl/IBackendObject.h"


namespace nbl::video
{

class IGPUSemaphore : public core::IReferenceCounted, public IBackendObject
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
        void* externalHandle = nullptr;
    };

    auto const& getCreationParams() const
    {
        return m_creationParams;
    }

    protected:
        IGPUSemaphore(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params = {})
            : IBackendObject(std::move(dev))
            , m_creationParams(std::move(params))
        {}

        virtual ~IGPUSemaphore() = default;

        // OpenGL: core::smart_refctd_ptr<COpenGLSync>*
        // Vulkan: const VkSemaphore*
        virtual void* getNativeHandle() = 0;
        SCreationParams m_creationParams;
};

}

#endif