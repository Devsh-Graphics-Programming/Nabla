#ifndef __NBL_I_GPU_SEMAPHORE_H_INCLUDED__
#define __NBL_I_GPU_SEMAPHORE_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"

namespace nbl {
namespace video
{

class IGPUSemaphore : public core::IReferenceCounted
{
protected:
    virtual ~IGPUSemaphore() = default;
};

}
}

#endif