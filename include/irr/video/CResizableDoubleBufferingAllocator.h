#ifndef __IRR_C_RESIZABLE_DOUBLE_BUFFERING_ALLOCATOR_H__
#define __IRR_C_RESIZABLE_DOUBLE_BUFFERING_ALLOCATOR_H__



#include "irr/video/CDoubleBufferingAllocator.h"

namespace irr
{
namespace video
{


template<class AddressAllocator, bool onlySwapRangesMarkedDirty = false, class CPUAllocator=core::allocator<uint8_t> >
class CResizableDoubleBufferingAllocator : public CDoubleBufferingAllocator<AddressAllocator,onlySwapRangesMarkedDirty,CPUAllocator>
{
    public:
        CResizableDoubleBufferingAllocator(video::IVideoDriver* driver, const IDriverMemoryBacked::SDriverMemoryRequirements& frontBufferReqs, const IDriverMemoryBacked::SDriverMemoryRequirements& backBufferReqs, auto growPolicy=, auto shrinkPolicy=)
        {
        }
        IMetaGranularGPUMappedBuffer(const size_t& granuleSize, const size_t& granuleCount, const bool& clientMemeory=true, const size_t& bufferGrowStep=512, const size_t& bufferShrinkStep=2048)
                                    : core::IMetaGranularBuffer<video::IGPUBuffer>(granuleSize,granuleCount,bufferGrowStep,bufferShrinkStep),
        IMetaGranularBuffer(const size_t& granuleSize, const size_t& granuleCount, const size_t& bufferGrowStep=512, const size_t& bufferShrinkStep=2048)
                            :   Allocated(0), Granules(granuleCount), GranuleByteSize(granuleSize), residencyRedirectTo(NULL),
                                BackBufferGrowStep(bufferGrowStep), BackBufferShrinkStep(bufferShrinkStep), B(NULL)
};

}
}

#endif // __IRR_C_RESIZABLE_DOUBLE_BUFFERING_ALLOCATOR_H__


