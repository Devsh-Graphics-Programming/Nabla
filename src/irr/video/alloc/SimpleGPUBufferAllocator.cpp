#include "irr/video/alloc/SimpleGPUBufferAllocator.h"
#include "IDriver.h"


using namespace irr;
using namespace video;


SimpleGPUBufferAllocator::value_type SimpleGPUBufferAllocator::allocate(size_t bytes, size_t alignment) noexcept
{
    auto reqs = mBufferMemReqs;
    reqs.vulkanReqs.size = bytes;
    reqs.vulkanReqs.alignment = alignment;
    auto buff = mDriver->createGPUBufferOnDedMem(reqs,false);
	buff->grab(); // don't want to be passing smart pointers around, this allocator it the only owner!
    return buff.get();
}
