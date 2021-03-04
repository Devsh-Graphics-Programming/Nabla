#ifndef __NBL_I_GPU_SEMAPHORE_H_INCLUDED__
#define __NBL_I_GPU_SEMAPHORE_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/video/IBackendObject.h"

namespace nbl {
namespace video
{

class IGPUSemaphore : public core::IReferenceCounted, public IBackendObject
{
protected:
    IGPUSemaphore(ILogicalDevice* dev) : IBackendObject(dev) {}

    virtual ~IGPUSemaphore() = default;
};

}
}

#endif