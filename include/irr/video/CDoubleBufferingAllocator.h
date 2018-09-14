#ifndef __IRR_C_DOUBLE_BUFFERING_ALLOCATOR_H__
#define __IRR_C_DOUBLE_BUFFERING_ALLOCATOR_H__


#include "IGPUBuffer.h"
#include "IDriverMemoryAllocation.h"
#include "IVideoDriver.h"

namespace irr
{
namespace video
{

template<class AddressAllocator, bool onlySwapRangesMarkedDirty>
class CDoubleBufferingAllocatorBase;
template<class AddressAllocator>
class CDoubleBufferingAllocatorBase<false>
{
    protected:
        std::pair<typename AddressAllocator::size_type,typename AddressAllocator::size_type> dirtyRange;

        CDoubleBufferingAllocatorBase() {}
        virtual ~CDoubleBufferingAllocatorBase() {}

        inline void resetDirtyRange() {}

        static constexpr bool alwaysSwapEntireRange = true;
};
template<class AddressAllocator>
class CDoubleBufferingAllocatorBase<true>
{
    protected:
        std::pair<typename AddressAllocator::size_type,typename AddressAllocator::size_type> dirtyRange;

        CDoubleBufferingAllocatorBase() {resetDirtyRange();}
        virtual ~CDoubleBufferingAllocatorBase() {}

        inline void resetDirtyRange() {dirtyRangeBegin = 0x7fffFFFFu; dirtyRangeEnd = 0;}

        static constexpr bool alwaysSwapEntireRange = false;
    public:
        inline void markRangeDirty(typename AddressAllocator::size_type begin, typename AddressAllocator::size_type end)
        {
            if (begin<dirtyRange.first) dirtyRange.first = begin;
            if (end>dirtyRange.second) dirtyRange.second = end;
        }
};


template<class AddressAllocator, bool onlySwapRangesMarkedDirty = false>
class CDoubleBufferingAllocator : public CDoubleBufferingAllocatorBase<AddressAllocator,onlySwapRangesMarkedDirty>, public virtual core::IReferenceCounted
{
    protected:
        IVideoDriver*       mDriver,
        IGPUBuffer*         mBackBuffer;
        IGPUBuffer*         mFrontBuffer;

        void*               mStagingPointer;
        size_t              mRangeStartRelToFB;
        size_t              mDestBuffOff;
        size_t              mReservedSize; // for allocator external state
        AddressAllocator    mAllocator;

        virtual ~CDoubleBufferingAllocator()
        {
            mBackBuffer->drop();
            mFrontBuffer->drop();
        }
    public:
		//! Creates a double buffering allocator from two GPU Buffers, where the staging buffer must have its bound memory already mapped with an appropriate range.
        /** Both buffers need to have already had their memory bound, i.e. IGPUBuffer::getBoundMemory cannot return nullptr.
        MEMORY CANNOT BE UNMAPPED OR REMAPPED FOR THE LIFETIME OF THE CDoubleBufferingAllocator OBJECT !!!
		@param rangeToUse is a memory range relative to the staging buffer's bound memory start, not the buffer start or the mapped IDriverMemoryAllocation offset. */
        template<typename... Args>
        CDoubleBufferingAllocator(IVideoDriver* driver, const IDriverMemoryAllocation::MemoryRange& rangeToUse, IGPUBuffer* stagingBuff, size_t destBuffOffset, IGPUBuffer* destBuff, Args&&... args) :
                        mDriver(driver), mBackBuffer(stagingBuff), mFrontBuffer(destBuff),
                        mStagingPointer(reinterpret_cast<uint8_t*>(mBackBuffer->getBoundMemory()->getMappedPointer())+rangeToUse.offset),
                        mRangeStartRelToFB(stagingBuff->getBoundMemoryOffset()-rangeToUse.offset), mDestBuffOff(destBuffOffset),
                        mReservedSize(AddressAllocator::reserved_size(reinterpret_cast<size_t>(mStagingPointer),bufSz,std::forward<Args>(args)...))
                        mAllocator(mStagingPointer,reinterpret_cast<size_t>(mStagingPointer)+mReservedSize,rangeToUse.length-mReservedSize)
        {
#ifdef _DEBUG
            assert(stagingBuff->getBoundMemoryOffset()>=rangeToUse.offset);
#endif // _DEBUG
            mBackBuffer->grab();
            mFrontBuffer->grab();
        }


        inline void* getBackBufferPointer() {return mStagingPointer;}

        inline IGPUBuffer* getFrontBuffer() {return mFrontBuffer;}


        template<typename... Args>
        inline typename AddressAllocator::size_type alloc_addr(std::forward<Args>(args)...) {return mAllocator.alloc_addr(std::forward<Args>(args)...);}

        template<typename... Args>
        inline typename AddressAllocator::size_type free_addr(std::forward<Args>(args)...) {return mAllocator.free_addr(std::forward<Args>(args)...);}


        //! Makes Writes visible
        inline void swapBuffers(void (*StuffToDoToFrontBuffer)(IGPUBuffer*,void*)=NULL,void* userData=NULL)
        {
            if (CDoubleBufferingAllocatorBase::alwaysSwapEntireRange)
                mDriver->copyBuffer(mBackBuffer,mFrontBuffer,mRangeStartRelToFB,mDestBuffOff,mEntireRange);
            else if (CDoubleBufferingAllocatorBase::dirtyRange.first<CDoubleBufferingAllocatorBase::dirtyRange.second)
            {
                mDriver->copyBuffer(mBackBuffer,mFrontBuffer,mRangeStartRelToFB,mDestBuffOff,
                                    CDoubleBufferingAllocatorBase::dirtyRange.second-CDoubleBufferingAllocatorBase::dirtyRange.first);
                CDoubleBufferingAllocatorBase::resetDirtyRange();
            }

            if (StuffToDoToFrontBuffer)
                StuffToDoToFrontBuffer(A,userData);
        }
};

}
}

#endif // __IRR_C_DOUBLE_BUFFERING_ALLOCATOR_H__

