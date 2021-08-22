#ifndef __NBL_I_DEFERRED_OPERATION_H_INCLUDED__
#define __NBL_I_DEFERRED_OPERATION_H_INCLUDED__


#include "nbl/core/IReferenceCounted.h"
#include "nbl/video/decl/IBackendObject.h"


namespace nbl::video
{

class IDeferredOperation : public core::IReferenceCounted, public IBackendObject
{
    public:
        enum E_STATUS : uint32_t
        {
            ES_COMPLETED,
            ES_NOT_READY,
            ES_THREAD_DONE,
            ES_THREAD_IDLE,
        };

        explicit IDeferredOperation(core::smart_refctd_ptr<const ILogicalDevice>&& dev) : IBackendObject(std::move(dev)) {}
        
        virtual bool join() = 0;
        virtual uint32_t getMaxConcurrency() = 0;
        virtual E_STATUS getStatus() = 0;
};

}

#endif