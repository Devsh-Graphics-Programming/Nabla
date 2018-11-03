#ifndef __IRR_GPU_MEMORY_ALLOCATOR_BASE_H__
#define __IRR_GPU_MEMORY_ALLOCATOR_BASE_H__

#include "IGPUBuffer.h"

namespace irr
{
namespace video
{

class IVideoDriver;

class GPUMemoryAllocatorBase
{
    protected:
        IVideoDriver*   mDriver;
        void            copyBuffersWrapper(IGPUBuffer* oldBuffer, IGPUBuffer* newBuffer, size_t oldOffset, size_t newOffset, size_t copyRangeLen);

        GPUMemoryAllocatorBase(IVideoDriver* inDriver) : mDriver(inDriver) {}
        virtual ~GPUMemoryAllocatorBase() {}
    public:
        size_t           min_alignment() const noexcept;

        IVideoDriver*    getDriver() noexcept;
};

}
}


#endif // __IRR_GPU_MEMORY_ALLOCATOR_BASE_H__
