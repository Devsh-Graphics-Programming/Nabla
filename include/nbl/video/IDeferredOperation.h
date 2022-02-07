#ifndef _NBL_VIDEO_I_DEFERRED_OPERATION_H_INCLUDED_
#define _NBL_VIDEO_I_DEFERRED_OPERATION_H_INCLUDED_

#include "nbl/core/IReferenceCounted.h"
#include "nbl/video/decl/IBackendObject.h"

namespace nbl::video
{
class IDeferredOperation : public core::IReferenceCounted, public IBackendObject
{
public:
    explicit IDeferredOperation(core::smart_refctd_ptr<const ILogicalDevice>&& dev)
        : IBackendObject(std::move(dev)) {}

public:
    enum E_STATUS : uint32_t
    {
        ES_COMPLETED,
        ES_NOT_READY,
        ES_THREAD_DONE,
        ES_THREAD_IDLE,
    };

    virtual bool join() = 0;
    virtual uint32_t getMaxConcurrency() = 0;
    virtual E_STATUS getStatus() = 0;
    virtual E_STATUS joinAndWait() = 0;
};

}

#endif