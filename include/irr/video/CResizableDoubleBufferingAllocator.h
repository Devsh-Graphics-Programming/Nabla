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
        typedef core::address_allocator_traits<AddressAllocator>                                            alloc_traits;

        typedef CDoubleBufferingAllocator<AddressAllocator,onlySwapRangesMarkedDirty,CPUAllocator>          Base;
        typedef typename AddressAllocator::size_type                                                        size_type;

        typedef CResizableDoubleBufferingAllocator<AddressAllocator,onlySwapRangesMarkedDirty,CPUAllocator> ThisType;


        inline void                             resizeBackBuffer(size_type newSize, size_type copyOldRangeLen)
        {
            IDriverMemoryAllocation* oldAlloc = const_cast<IDriverMemoryAllocation*>(Base::mBackBuffer->getBoundMemory());
            //! TODO: Use a GPU heap allocator instead of buffer directly -- after move to Vulkan only
            IDriverMemoryBacked::SDriverMemoryRequirements reqs = Base::mBackBuffer->getMemoryReqs();
            reqs.vulkanReqs.size = Base::mStagingBuffOff+newSize;

            //! No BoundMemoryOffset applied to mStagingBuffOff on new buffer as allocated on dedicated memory
            IGPUBuffer* rep = Base::mDriver->createGPUBufferOnDedMem(reqs,Base::mBackBuffer->canUpdateSubRange());
            // ignore return value as it has wrong offsets
            const_cast<IDriverMemoryAllocation*>(rep->getBoundMemory())->mapMemoryRange(oldAlloc->getCurrentMappingCaps(),
                                                        IDriverMemoryAllocation::MemoryRange{Base::mStagingBuffOff,newSize});
            {
                uint8_t* newMappedMem = reinterpret_cast<uint8_t*>(rep->getBoundMemory()->getMappedPointer())+Base::mStagingBuffOff;
                memcpy(newMappedMem,Base::mStagingPointer,copyOldRangeLen);
                Base::mRangeLength = newSize;
                Base::mStagingPointer = newMappedMem;
            }
            oldAlloc->unmapMemory();
            Base::mBackBuffer->pseudoMoveAssign(rep);
            rep->drop();
        }
        inline void                             resizeFrontBuffer(size_type newSize, size_type copyOldRangeLen)
        {
            //! TODO: Use a GPU heap allocator instead of buffer directly -- after move to Vulkan only
            IDriverMemoryBacked::SDriverMemoryRequirements reqs = Base::mFrontBuffer->getMemoryReqs();
            reqs.vulkanReqs.size = Base::mDestBuffOff+newSize;

            IGPUBuffer* newFrontBuffer = Base::mDriver->createGPUBufferOnDedMem(reqs,Base::mFrontBuffer->canUpdateSubRange());
            Base::mDriver->copyBuffer(Base::mFrontBuffer,newFrontBuffer,Base::mDestBuffOff,Base::mDestBuffOff,copyOldRangeLen);
            Base::mFrontBuffer->pseudoMoveAssign(newFrontBuffer);
            newFrontBuffer->drop();
        }

    protected:
        ///constexpr static size_type growStep = 2u; //debug test
        constexpr static size_type growStep = 32u*4096u; //128k at a time
        constexpr static size_type growStepMinus1 = growStep-1u;
        static inline size_type                 defaultGrowPolicy(ThisType* _this, size_type totalRequestedNewMem)
        {
            size_type allAllocatorSpace = alloc_traits::get_total_size(_this->mAllocator);
            size_type nextAllocTotal = alloc_traits::get_allocated_size(_this->mAllocator)+totalRequestedNewMem;
            if (nextAllocTotal>allAllocatorSpace)
                return (nextAllocTotal+growStepMinus1)&(~growStepMinus1);

            return allAllocatorSpace;
        }
        static inline size_type                 defaultShrinkPolicy(ThisType* _this)
        {
            ///constexpr size_type shrinkStep = 2u; //debug test
            constexpr size_type shrinkStep = 256u*4096u; //1M at a time

            size_type allFreeSpace = alloc_traits::get_free_size(_this->mAllocator);
            if (allFreeSpace>shrinkStep)
                return (alloc_traits::get_allocated_size(_this->mAllocator)+growStepMinus1)&(~growStepMinus1);

            return alloc_traits::get_total_size(_this->mAllocator);
        }

        size_type(*growPolicy)(ThisType*,size_type);
        size_type(*shrinkPolicy)(ThisType*);

    public:
        static inline size_type                 calcBufferAlignPadding(size_type newSize, IVideoDriver* driver, size_type maxReqAlignment)
        {
            size_type defaultAlignment = driver->getMinimumMemoryMapAlignment();
            if (defaultAlignment>maxReqAlignment)
                return 0u;

            size_type worstPossibleStartAlignment = maxReqAlignment-defaultAlignment;
            return worstPossibleStartAlignment;
        }

        //! The IDriverMemoryBacked::SDriverMemoryRequirements::size must be identical for both reqs
        template<typename... Args>
        CResizableDoubleBufferingAllocator( IVideoDriver* driver, const IDriverMemoryBacked::SDriverMemoryRequirements& stagingBufferReqs,
                                            const IDriverMemoryBacked::SDriverMemoryRequirements& frontBufferReqs, Args&&... args) : // delegate
                                            CResizableDoubleBufferingAllocator(driver,
                                                                               IDriverMemoryAllocation::MemoryRange(0,stagingBufferReqs.vulkanReqs.size),
                                                                               driver->createGPUBufferOnDedMem(stagingBufferReqs,false), 0u,
                                                                               driver->createGPUBufferOnDedMem(frontBufferReqs,false), std::forward<Args>(args)...)
        {
#ifdef _DEBUG
            assert(stagingBufferReqs.vulkanReqs.size==frontBufferReqs.vulkanReqs.size);
#endif // _DEBUG
        }

        //! DO NOT USE THIS CONTRUCTOR UNLESS YOU MEAN FOR THE IGPUBufferS TO BE RESIZED AT WILL !
        template<typename... Args>
        CResizableDoubleBufferingAllocator( IVideoDriver* driver, const IDriverMemoryAllocation::MemoryRange& rangeToUse, IGPUBuffer* stagingBuff,
                                            size_t destBuffOffset, IGPUBuffer* destBuff, Args&&... args) :
                                                Base(driver,rangeToUse,stagingBuff,destBuffOffset,destBuff,std::forward<Args>(args)...),
                                                            growPolicy(defaultGrowPolicy), shrinkPolicy(defaultShrinkPolicy)
        {
        }

        //! Grow Policies return
        inline const decltype(growPolicy)&      getGrowPolicy() const {return growPolicy;}
        inline void                             setGrowPolicy(const decltype(growPolicy)& newGrowPolicy) {growPolicy=newGrowPolicy;}

        inline const decltype(shrinkPolicy)&    getShrinkPolicy() const {return shrinkPolicy;}
        inline void                             setShrinkPolicy(const decltype(shrinkPolicy)& newShrinkPolicy) {shrinkPolicy=newShrinkPolicy;}


        template<typename... Args>
        inline void                             multi_alloc_addr(size_type* outAddresses, uint32_t count, const size_type* bytes, Args&&... args)
        {
            size_type maxRequestedAllocSize = bytes[0];
            size_type totalRequestedNewMem = bytes[0];
            for (uint32_t i=1; i<count; i++)
            {
                totalRequestedNewMem += bytes[i];
                if (bytes[i]>maxRequestedAllocSize)
                    maxRequestedAllocSize = bytes[i];
            }

            size_type allAllocatorSpace = alloc_traits::get_total_size(Base::mAllocator);
            size_type newSize = growPolicy(this,totalRequestedNewMem);
            newSize = std::max(newSize,alloc_traits::get_allocated_size(Base::mAllocator)+totalRequestedNewMem);
            //! TODO: Handle cases when fragmentation requires much more new space to be added
            //if (newSize <=>?! maxRequestedAllocSize)
                //newSize = allAllocatorSpace+totalRequestedNewMem;

            if (newSize>allAllocatorSpace)
            {
                newSize += calcBufferAlignPadding(newSize,Base::mDriver,Base::mAllocator.max_alignment());

                void* newAllocatorState = Base::mAllocatorState;
                size_type newReservedSize = AddressAllocator::reserved_size(newSize,Base::mAllocator);
                if (newReservedSize>Base::mReservedSize)
                    newAllocatorState = Base::mCPUAllocator.allocate(newReservedSize,_IRR_SIMD_ALIGNMENT);

                size_type oldRangeLength = Base::mRangeLength;
                if (Base::mStagingBuffOff+newSize>Base::mBackBuffer->getSize())
                    resizeBackBuffer(newSize,oldRangeLength);

                Base::mAllocator = AddressAllocator(Base::mAllocator,newAllocatorState,Base::mStagingPointer,newSize); //! handle offset padding impact on newSize
                if (newReservedSize>Base::mReservedSize) // equivalent to (newAllocatorState!=mAllocatorState)
                {
                    Base::mCPUAllocator.deallocate(reinterpret_cast<uint8_t*>(Base::mAllocatorState),Base::mReservedSize);
                    Base::mAllocatorState = newAllocatorState;
                    Base::mReservedSize = newReservedSize;
                }


                if (Base::mDestBuffOff+newSize>Base::mFrontBuffer->getSize())
                    resizeFrontBuffer(newSize,oldRangeLength);
            }

            alloc_traits::multi_alloc_addr(Base::mAllocator,outAddresses,count,bytes,std::forward<Args>(args)...);
        }

        template<typename... Args>
        inline void                             multi_free_addr(Args&&... args)
        {
            alloc_traits::multi_free_addr(Base::mAllocator,std::forward<Args>(args)...);

            size_type allAllocatorSpace = alloc_traits::get_total_size(Base::mAllocator);
            size_type newSize = shrinkPolicy(this);
            if (newSize>=allAllocatorSpace)
                return;

            // some allocators may not be shrinkable because of fragmentation
            newSize = Base::mAllocator.safe_shrink_size(newSize);
            if (newSize>=allAllocatorSpace)
                return;
            // or alignment
            newSize += calcBufferAlignPadding(newSize,Base::mDriver,Base::mAllocator.max_alignment());
            if (newSize>=allAllocatorSpace)
                return;

            //! don't bother resizing reserved space

            // resize back buffer and allocator
            if (Base::mStagingBuffOff+newSize<Base::mBackBuffer->getSize())
                resizeBackBuffer(newSize,newSize);

            Base::mAllocator = AddressAllocator(Base::mAllocator,Base::mAllocatorState,Base::mStagingPointer,newSize); //! handle offset padding impact on newSize

            // resize front buffer
            if (Base::mDestBuffOff+newSize<Base::mFrontBuffer->getSize())
                resizeFrontBuffer(newSize,newSize);
        }
};

}
}

#endif // __IRR_C_RESIZABLE_DOUBLE_BUFFERING_ALLOCATOR_H__


