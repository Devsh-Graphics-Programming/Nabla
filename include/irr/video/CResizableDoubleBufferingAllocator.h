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
    private:
        typedef CDoubleBufferingAllocator<AddressAllocator,onlySwapRangesMarkedDirty,CPUAllocator>  Base;
    protected:
        auto growPolicy;
        auto shrinkPolicy;
    public:
        //! The IDriverMemoryBacked::SDriverMemoryRequirements::size must be identical for both reqs
        template<typename... Args>
        CResizableDoubleBufferingAllocator( video::IVideoDriver* driver,
                                            const IDriverMemoryBacked::SDriverMemoryRequirements& stagingBufferReqs,
                                            const IDriverMemoryBacked::SDriverMemoryRequirements& frontBufferReqs, Args&&... args) : // delegate
                                            CResizableDoubleBufferingAllocator(driver,
                                                                               IDriverMemoryAllocation::MemoryRange(0,stagingBufferReqs.vulkanReqs.size),
                                                                               stagingBuffer,
                                                                               0u, frontBuffer, std::forward<Args>(args)...)
        {
#ifdef _DEBUG
            assert(stagingBufferReqs.vulkanReqs.size--frontBufferReqs.vulkanReqs.size);
#endif // _DEBUG
        }

        //! DO NOT USE THIS CONTRUCTOR UNLESS THE IGPUBufferS ARE MEANT TO BE RESIZED AT WILL !
        template<typename... Args>
        CResizableDoubleBufferingAllocator( video::IVideoDriver* driver, const IDriverMemoryAllocation::MemoryRange& rangeToUse, IGPUBuffer* stagingBuff,
                                            size_t destBuffOffset, IGPUBuffer* destBuff, Args&&... args) :
                                                Base(driver,rangeToUse,stagingBuff,destBuffOffset,destBuff,std::forward<Args>(args)...)
        {
            growPolicy = ;
            shrinkPolicy = ;
        }


        inline const auto&  getGrowPolicy() const {return growPolicy;}
        inline void         setGrowPolicy(const decltype(growPolicy)& newGrowPolicy) {growPolicy=newGrowPolicy;}

        inline const auto&  getShrinkPolicy() const {return shrinkPolicy;}
        inline void         setShrinkPolicy(const decltype(shrinkPolicy)& newShrinkPolicy) {shrinkPolicy=newShrinkPolicy;}


        template<typename... Args>
        inline void                     multi_alloc_addr(Args&&... args)
        {
            size_type reserveSize = growPolicy(this,);
            if (reserveSize+Base::mOffset>Base::mBackBuffer->getBoundMemory()->getAllocationSize())
                swapstuff;

            alloc_traits::multi_alloc_addr(mAllocator,std::forward<Args>(args)...);
        }

        template<typename... Args>
        inline void                     multi_free_addr(Args&&... args)
        {
            alloc_traits::multi_free_addr(mAllocator,std::forward<Args>(args)...);
        }
};

}
}

#endif // __IRR_C_RESIZABLE_DOUBLE_BUFFERING_ALLOCATOR_H__


