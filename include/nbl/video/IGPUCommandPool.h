#ifndef __NBL_I_GPU_COMMAND_POOL_H_INCLUDED__
#define __NBL_I_GPU_COMMAND_POOL_H_INCLUDED__


#include "nbl/core/IReferenceCounted.h"

#include "nbl/video/decl/IBackendObject.h"


namespace nbl::video
{

class IGPUCommandPool : public core::IReferenceCounted, public IBackendObject
{
    public:
        enum E_CREATE_FLAGS : uint32_t
        {
            ECF_TRANSIENT_BIT = 0x01,
            ECF_RESET_COMMAND_BUFFER_BIT = 0x02,
            ECF_PROTECTED_BIT = 0x04
        };

        IGPUCommandPool(core::smart_refctd_ptr<const ILogicalDevice>&& dev, E_CREATE_FLAGS _flags, uint32_t _familyIx) : IBackendObject(std::move(dev)), m_flags(_flags), m_familyIx(_familyIx) {}

        E_CREATE_FLAGS getCreationFlags() const { return m_flags; }
        uint32_t getQueueFamilyIndex() const { return m_familyIx; }

    protected:
        virtual ~IGPUCommandPool() = default;

        E_CREATE_FLAGS m_flags;
        uint32_t m_familyIx;
};

}


#endif