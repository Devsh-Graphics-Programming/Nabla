#ifndef __IRR_C_DOUBLE_BUFFERING_ALLOCATOR_H__
#define __IRR_C_DOUBLE_BUFFERING_ALLOCATOR_H__


#include "IGPUBuffer.h"
#include "IDriverMemoryAllocation.h"
#include "IVideoDriver.h"

#include "irr/core/alloc/address_allocator_traits.h"

namespace irr
{
namespace video
{


template<class AddressAllocator, bool onlySwapRangesMarkedDirty = false, class CPUAllocator=core::allocator<uint8_t> >
class CDoubleBufferingAllocator : public core::MultiBufferingAllocatorBase<AddressAllocator,onlySwapRangesMarkedDirty>, public virtual core::IReferenceCounted
{
    private:
        typedef core::MultiBufferingAllocatorBase<AddressAllocator,onlySwapRangesMarkedDirty>   Base;

        static inline typename AddressAllocator::size_type calcConservBuffAlign(IVideoDriver* driver, void* ptr)
        {
            typename AddressAllocator::size_type baseAlign = driver->getMinimumMemoryMapAlignment();
            typename AddressAllocator::size_type offsetAlign = 0x1u<<core::findLSB(reinterpret_cast<size_t>(ptr));
            return std::min(baseAlign,offsetAlign);
        }
    protected:
        IVideoDriver*       mDriver;
        IGPUBuffer*         mBackBuffer;
        IGPUBuffer*         mFrontBuffer;
        size_t              mStagingBuffOff;
        size_t              mRangeLength;
        void*               mStagingPointer;

        size_t              mDestBuffOff;

        size_t              mReservedSize; // for allocator external state
        CPUAllocator        mCPUAllocator;
        void*               mAllocatorState;
        AddressAllocator    mAllocator;

        virtual ~CDoubleBufferingAllocator()
        {
            mCPUAllocator.deallocate(reinterpret_cast<uint8_t*>(mAllocatorState),mReservedSize);
            mBackBuffer->drop();
            mFrontBuffer->drop();
        }
    public:
        typedef core::address_allocator_traits<AddressAllocator>                            alloc_traits;

		//! Creates a double buffering allocator from two GPU Buffers, where the staging buffer must have its bound memory already mapped with an appropriate range.
        /** Both buffers need to have already had their memory bound, i.e. IGPUBuffer::getBoundMemory cannot return nullptr.
        MEMORY CANNOT BE UNMAPPED OR REMAPPED FOR THE LIFETIME OF THE CDoubleBufferingAllocator OBJECT !!!
		@param rangeToUse is a memory range relative to the staging buffer's bound memory start, not the buffer start or the mapped IDriverMemoryAllocation offset. */
        template<typename... Args>
        CDoubleBufferingAllocator(IVideoDriver* driver, const IDriverMemoryAllocation::MemoryRange& rangeToUse, IGPUBuffer* stagingBuff, size_t destBuffOffset, IGPUBuffer* destBuff, Args&&... args) :
                        mDriver(driver), mBackBuffer(stagingBuff), mFrontBuffer(destBuff), mStagingBuffOff(rangeToUse.offset), mRangeLength(rangeToUse.length),
                        mStagingPointer(reinterpret_cast<uint8_t*>(mBackBuffer->getBoundMemory()->getMappedPointer())+mBackBuffer->getBoundMemoryOffset()+mStagingBuffOff),
                        mDestBuffOff(destBuffOffset), mReservedSize(AddressAllocator::reserved_size(mRangeLength,calcConservBuffAlign(mDriver,mStagingPointer),args...)), mCPUAllocator(),
                        mAllocatorState(mCPUAllocator.allocate(mReservedSize,_IRR_SIMD_ALIGNMENT)),
                        mAllocator(mAllocatorState,mStagingPointer,calcConservBuffAlign(mDriver,mStagingPointer),mRangeLength,std::forward<Args>(args)...)
        {
#ifdef _DEBUG
            assert(mBackBuffer->getBoundMemoryOffset()+mStagingBuffOff+mRangeLength>=mBackBuffer->getBoundMemory()->getAllocationSize());
#endif // _DEBUG
            mBackBuffer->grab();
            mFrontBuffer->grab();
        }

        inline const AddressAllocator&  getAllocator() const {return mAllocator;}

        inline void*                    getBackBufferPointer() {return mStagingPointer;}

        inline IGPUBuffer*              getFrontBuffer() {return mFrontBuffer;}


        template<typename... Args>
        inline void                     multi_alloc_addr(Args&&... args) {alloc_traits::multi_alloc_addr(mAllocator,std::forward<Args>(args)...);}

        template<typename... Args>
        inline void                     multi_free_addr(Args&&... args) {alloc_traits::multi_free_addr(mAllocator,std::forward<Args>(args)...);}


        //! Makes Writes visible
        inline void                     swapBuffers(void (*StuffToDoToBackBuffer)(IGPUBuffer*,void*)=NULL, void (*StuffToDoToFrontBuffer)(IGPUBuffer*,void*)=NULL,void* userData=NULL)
        {
            if (StuffToDoToBackBuffer)
                StuffToDoToBackBuffer(mBackBuffer,userData);

            if (Base::alwaysSwapEntireRange)
            {
                if (mBackBuffer->getBoundMemory()->haveToFlushWrites())
                {
                    IDriverMemoryAllocation::MappedMemoryRange range(mBackBuffer->getBoundMemory(),mStagingBuffOff+mBackBuffer->getBoundMemoryOffset(),mRangeLength);
                    mDriver->flushMappedMemoryRanges(1,&range);
                }

                mDriver->copyBuffer(mBackBuffer,mFrontBuffer,mStagingBuffOff,mDestBuffOff,mRangeLength);
            }
            else if (Base::dirtyRange.first<Base::dirtyRange.second)
            {
                typename AddressAllocator::size_type dirtyRangeLen = Base::dirtyRange.second-Base::dirtyRange.first;
                if (mBackBuffer->getBoundMemory()->haveToFlushWrites())
                {
                    IDriverMemoryAllocation::MappedMemoryRange range(mBackBuffer->getBoundMemory(),mStagingBuffOff+mBackBuffer->getBoundMemoryOffset()+Base::dirtyRange.first,dirtyRangeLen);
                    mDriver->flushMappedMemoryRanges(1,&range);
                }

                mDriver->copyBuffer(mBackBuffer,mFrontBuffer,mStagingBuffOff+Base::dirtyRange.first,
                                    mDestBuffOff+Base::dirtyRange.first,dirtyRangeLen);
                Base::resetDirtyRange();
            }

            if (StuffToDoToFrontBuffer)
                StuffToDoToFrontBuffer(mFrontBuffer,userData);
        }
};

}
}

#endif // __IRR_C_DOUBLE_BUFFERING_ALLOCATOR_H__

