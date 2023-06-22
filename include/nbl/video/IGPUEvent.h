#ifndef _NBL_VIDEO_I_GPU_EVENT_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_EVENT_H_INCLUDED_


#include "nbl/core/IReferenceCounted.h"

#include "nbl/asset/IEvent.h"

#include "nbl/video/decl/IBackendObject.h"


namespace nbl::video
{

class IGPUEvent : public core::IReferenceCounted, public asset::IEvent, public IBackendObject
{
    protected:
        explicit IGPUEvent(core::smart_refctd_ptr<const ILogicalDevice>&& dev, CREATE_FLAGS flags) : asset::IEvent(flags), IBackendObject(std::move(dev)) {}
        virtual ~IGPUEvent() = default;
};

}

#endif
