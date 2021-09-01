#ifndef __NBL_I_DESCRIPTOR_POOL_H_INCLUDED__
#define __NBL_I_DESCRIPTOR_POOL_H_INCLUDED__


#include "nbl/core/IReferenceCounted.h"

#include "nbl/asset/IDescriptorSetLayout.h"

#include "nbl/video/decl/IBackendObject.h"


namespace nbl::video
{

class IDescriptorPool : public core::IReferenceCounted, public IBackendObject
{
    public:
        enum E_CREATE_FLAGS : uint32_t
        {
            ECF_NONE = 0x00u,
            ECF_FREE_DESCRIPTOR_SET_BIT = 0x01,
            ECF_UPDATE_AFTER_BIND_BIT = 0x02,
            ECF_HOST_ONLY_BIT_VALVE = 0x04
        };

        struct SDescriptorPoolSize
        {
            asset::E_DESCRIPTOR_TYPE type;
            uint32_t count;
        };

        explicit IDescriptorPool(core::smart_refctd_ptr<const ILogicalDevice>&& dev, uint32_t _maxSets) : IBackendObject(std::move(dev)), m_maxSets(_maxSets) {}

        uint32_t getCapacity() const { return m_maxSets; }

    protected:
        uint32_t m_maxSets;
};

}

#endif