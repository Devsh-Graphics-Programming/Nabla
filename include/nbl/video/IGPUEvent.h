#ifndef __NBL_I_GPU_EVENT_H_INCLUDED__
#define __NBL_I_GPU_EVENT_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/asset/IEvent.h"
#include "nbl/video/IBackendObject.h"

namespace nbl {
namespace video
{

class IGPUEvent : public asset::IEvent, public core::IReferenceCounted, public IBackendObject
{
public:
    explicit IGPUEvent(ILogicalDevice* dev, E_CREATE_FLAGS flags) : asset::IEvent(flags), IBackendObject(dev) 
    {
        assert(flags&ECF_DEVICE_ONLY_BIT); // for now allow only DEVICE_ONLY events
    }

protected:
    virtual ~IGPUEvent() = default;
};

}
}

#endif
