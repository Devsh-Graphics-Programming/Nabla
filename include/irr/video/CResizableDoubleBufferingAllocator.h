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
        typedef CResizableDoubleBufferingAllocator<AddressAllocator,onlySwapRangesMarkedDirty,CPUAllocator>  ThisType;

    protected:
        static size_type defaultGrowPolicy(ThisType* _this, size_type totalRequestedNewMem)
        {
            constexpr size_type growStep = 32u*4096u; //128k at a time
            return (_this->()+totalRequestedNewMem+growStep)&size_type(growStep-1u);
        }
        static size_type defaultShrinkPolicy(ThisType* _this,)
        {
            return stuffs;
        }

        decltype(defaultGrowPolicy) growPolicy;
        decltype(defaultShrinkPolicy) shrinkPolicy;

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
            assert(stagingBufferReqs.vulkanReqs.size==frontBufferReqs.vulkanReqs.size);
#endif // _DEBUG
        }

        //! DO NOT USE THIS CONTRUCTOR UNLESS YOU MEAN FOR THE IGPUBufferS TO BE RESIZED AT WILL !
        template<typename... Args>
        CResizableDoubleBufferingAllocator( video::IVideoDriver* driver, const IDriverMemoryAllocation::MemoryRange& rangeToUse, IGPUBuffer* stagingBuff,
                                            size_t destBuffOffset, IGPUBuffer* destBuff, Args&&... args) :
                                                Base(driver,rangeToUse,stagingBuff,destBuffOffset,destBuff,std::forward<Args>(args)...),
                                                            growPolicy(defaultGrowPolicy), shrinkPolicy(defaultShrinkPolicy)
        {
        }

        //! Grow Policies return
        inline const auto&  getGrowPolicy() const {return growPolicy;}
        inline void         setGrowPolicy(const decltype(growPolicy)& newGrowPolicy) {growPolicy=newGrowPolicy;}

        inline const auto&  getShrinkPolicy() const {return shrinkPolicy;}
        inline void         setShrinkPolicy(const decltype(shrinkPolicy)& newShrinkPolicy) {shrinkPolicy=newShrinkPolicy;}


        template<typename... Args>
        inline void                     multi_alloc_addr(size_type* outAddresses, uint32_t count, const size_type* bytes, Args&&... args)
        {
            size_type totalRequestedNewMem = bytes[0];
            for (uint32_t i=1; i<count; i++)
                totalRequestedNewMem += bytes[i];

            size_type reserveSize = growPolicy(this,totalRequestedNewMem);
            if (reserveSize+Base::mOffset>Base::mBackBuffer->getBoundMemory()->getAllocationSize())
                swapstuff;

            alloc_traits::multi_alloc_addr(mAllocator,std::forward<Args>(args)...);
        }

        template<typename... Args>
        inline void                     multi_free_addr(Args&&... args)
        {
            alloc_traits::multi_free_addr(mAllocator,std::forward<Args>(args)...);

            size_type reserveSize = shrinkPolicy(this,);
            if (reserveSize!=)
            {
                reserveSize = mAllocator.safe_shrink_size(reserveSize);
                if (reserveSize==)
                    return;

                // resize
            }

        }
};

}
}

#endif // __IRR_C_RESIZABLE_DOUBLE_BUFFERING_ALLOCATOR_H__


