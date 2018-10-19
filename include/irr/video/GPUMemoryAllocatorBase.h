#ifndef __IRR_GPU_MEMORY_ALLOCATOR_BASE_H__
#define __IRR_GPU_MEMORY_ALLOCATOR_BASE_H__

#include "IVideoDriver.h"


namespace irr
{
namespace video
{

class GPUMemoryAllocatorBase
{
    protected:
        IVideoDriver* mDriver;

        GPUMemoryAllocatorBase(IVideoDriver* inDriver) : mDriver(inDriver) {}
        virtual ~GPUMemoryAllocatorBase() {}
    public:
        inline size_t           min_alignment() const
        {
            return mDriver->getMinimumMemoryMapAlignment();
        }

        inline IVideoDriver*    getDriver() noexcept
        {
            return mDriver;
        }
};

}
}


#endif // __IRR_GPU_MEMORY_ALLOCATOR_BASE_H__
