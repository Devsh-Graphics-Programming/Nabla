#ifndef __IRR_GPU_MEMORY_ALLOCATOR_BASE_H__
#define __IRR_GPU_MEMORY_ALLOCATOR_BASE_H__

#include "IGPUBuffer.h"

namespace irr
{
namespace video
{

class IDriver;

class GPUMemoryAllocatorBase
{
    protected:
        IDriver*   mDriver;
        void            copyBuffersWrapper(IGPUBuffer* oldBuffer, IGPUBuffer* newBuffer, size_t oldOffset, size_t newOffset, size_t copyRangeLen);

        GPUMemoryAllocatorBase(IDriver* inDriver) : mDriver(inDriver) {}
        virtual ~GPUMemoryAllocatorBase() {}
    public:
        IDriver*    getDriver() noexcept {return mDriver;}
};

}
}


#endif // __IRR_GPU_MEMORY_ALLOCATOR_BASE_H__
