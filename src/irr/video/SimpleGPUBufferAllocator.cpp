#include <cstring>

#include "irr/video/SimpleGPUBufferAllocator.h"
#include "IVideoDriver.h"


using namespace irr;
using namespace video;


IGPUBuffer* SimpleGPUBufferAllocator::createBuffer()
{
    return mDriver->createGPUBufferOnDedMem(mBufferMemReqs,false);
}

