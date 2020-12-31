#ifndef __NBL_I_GPU_EVENT_H_INCLUDED__
#define __NBL_I_GPU_EVENT_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/asset/IEvent.h"

namespace nbl {
namespace video
{

class IGPUEvent : public asset::IEvent, public core::IReferenceCounted
{
public:
    using asset::IEvent::IEvent;

protected:
    virtual ~IGPUEvent() = default;
};

}
}

#endif
