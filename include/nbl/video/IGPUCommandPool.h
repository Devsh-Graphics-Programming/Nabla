#ifndef __NBL_I_GPU_COMMAND_POOL_H_INCLUDED__
#define __NBL_I_GPU_COMMAND_POOL_H_INCLUDED__


#include "nbl/core/IReferenceCounted.h"
#include "nbl/core/util/bitflag.h"

#include "nbl/video/decl/IBackendObject.h"


namespace nbl::video
{

class IGPUCommandPool : public core::IReferenceCounted, public IBackendObject
{
    public:
        enum E_CREATE_FLAGS : uint32_t
        {
            ECF_NONE = 0x00,
            ECF_TRANSIENT_BIT = 0x01,
            ECF_RESET_COMMAND_BUFFER_BIT = 0x02,
            ECF_PROTECTED_BIT = 0x04
        };

        IGPUCommandPool(core::smart_refctd_ptr<const ILogicalDevice>&& dev, core::bitflag<E_CREATE_FLAGS> _flags, uint32_t _familyIx) : IBackendObject(std::move(dev)), m_flags(_flags), m_familyIx(_familyIx) {}

        core::bitflag<E_CREATE_FLAGS> getCreationFlags() const { return m_flags; }
        uint32_t getQueueFamilyIndex() const { return m_familyIx; }

        // OpenGL: nullptr, because commandpool doesn't exist in GL (we might expose the internal allocator in the future)
        // Vulkan: const VkCommandPool*
        virtual const void* getNativeHandle() const = 0;

    protected:
        virtual ~IGPUCommandPool() = default;

        core::bitflag<E_CREATE_FLAGS> m_flags;
        uint32_t m_familyIx;
};

}


#endif