#ifndef __IRR_SIMPLE_GPU_BUFFER_ALLOCATOR_H__
#define __IRR_SIMPLE_GPU_BUFFER_ALLOCATOR_H__

#include "IGPUBuffer.h"
#include "irr/video/GPUMemoryAllocatorBase.h"


namespace irr
{
namespace video
{

namespace impl
{
    class SimpleGPUBufferAllocatorBase : public GPUMemoryAllocatorBase
    {
        protected:
            IGPUBuffer* createGPUBuffer(const IDriverMemoryBacked::SDriverMemoryRequirements& bufferMemReqs);
        public:
            using GPUMemoryAllocatorBase::GPUMemoryAllocatorBase;
    };
}

//! TODO: actual implementation of a SimpleGPUBufferAllocator

}
}

#include "IVideoDriver.h"
namespace irr
{
namespace video
{
// inlines
}
}

#endif // __IRR_SIMPLE_GPU_BUFFER_ALLOCATOR_H__

