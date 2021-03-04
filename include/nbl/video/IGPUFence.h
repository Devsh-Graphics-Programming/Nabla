#ifndef __NBL_I_GPU_FENCE_H_INCLUDED__
#define __NBL_I_GPU_FENCE_H_INCLUDED__

#include <nbl/core/IReferenceCounted.h>
#include "nbl/video/IBackendObject.h"

namespace nbl {
namespace video
{

class IGPUFence : public core::IReferenceCounted, public IBackendObject
{
public:
    enum E_CREATE_FLAGS : uint32_t
    {
        ECF_SIGNALED_BIT = 0x01u
    };
    enum E_STATUS
    {
        ES_SUCCESS,
        ES_TIMEOUT,
        ES_NOT_READY,
        ES_ERROR
    };

    IGPUFence(ILogicalDevice* dev, E_CREATE_FLAGS flags) : IBackendObject(dev), m_status((flags&ECF_SIGNALED_BIT) ? ES_SUCCESS:ES_NOT_READY)
    {
    }

protected:
    virtual ~IGPUFence() = default;

    inline E_STATUS setStatus(E_STATUS s) { return m_status = s; }

    mutable E_STATUS m_status;
};

}
}

#endif