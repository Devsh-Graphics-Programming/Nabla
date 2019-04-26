// Copyright (C) 2018 Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW Engine"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_POOL_ADDRESS_ALLOCATOR_H_INCLUDED__
#define __IRR_POOL_ADDRESS_ALLOCATOR_H_INCLUDED__

#include "IrrCompileConfig.h"

#include "irr/core/math/irrMath.h"

#include "irr/core/alloc/AddressAllocatorBase.h"

namespace irr
{
namespace core
{


//! Can only allocate up to a size of a single block, no support for allocations larger than blocksize
template<typename _size_type>
class PoolAddressAllocator : public AddressAllocatorBase<PoolAddressAllocator<_size_type>,_size_type>
{
    private:
        typedef AddressAllocatorBase<PoolAddressAllocator<_size_type>,_size_type> Base;
    public:
        _IRR_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(_size_type);

        static constexpr bool supportsNullBuffer = true;

        #define DUMMY_DEFAULT_CONSTRUCTOR PoolAddressAllocator() : blockSize(1u), blockCount(0u) {}
        GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(DUMMY_DEFAULT_CONSTRUCTOR)
        #undef DUMMY_DEFAULT_CONSTRUCTOR

        virtual ~PoolAddressAllocator() {}

        PoolAddressAllocator(void* reservedSpc, _size_type addressOffsetToApply, _size_type alignOffsetNeeded, _size_type maxAllocatableAlignment, size_type bufSz, size_type blockSz) noexcept :
					Base(reservedSpc,addressOffsetToApply,alignOffsetNeeded,maxAllocatableAlignment),
						blockCount((bufSz-alignOffsetNeeded)/blockSz), blockSize(blockSz), freeStack(reinterpret_cast<size_type*>(Base::reservedSpace)), freeStackCtr(0u)
        {
            reset();
        }

        //! When resizing we require that the copying of data buffer has already been handled by the user of the address allocator even if `supportsNullBuffer==true`
        template<typename... Args>
        PoolAddressAllocator(_size_type newBuffSz, PoolAddressAllocator&& other, Args&&... args) noexcept :
					Base(std::move(other),std::forward<Args>(args)...),
						blockCount((newBuffSz-Base::alignOffset)/other.blockSize), blockSize(other.blockSize), freeStack(reinterpret_cast<size_type*>(Base::reservedSpace)), freeStackCtr(0u)
        {
            if (blockCount>other.blockCount)
                freeStackCtr = blockCount-other.blockCount;
            other.blockCount = invalid_address;

            #ifdef _IRR_DEBUG
                assert(Base::checkResize(newBuffSz,Base::alignOffset));
            #endif // _IRR_DEBUG
			other.blockSize = invalid_address;

            for (size_type i=0u; i<freeStackCtr; i++)
                freeStack[i] = (blockCount-1u-i)*blockSize+Base::combinedOffset;

            for (size_type i=0; i<other.freeStackCtr; i++)
            {
                size_type freeEntry = other.freeStack[i]-other.combinedOffset;

                if (freeEntry<blockCount*blockSize)
                    freeStack[freeStackCtr++] = freeEntry+Base::combinedOffset;
            }
            other.freeStackCtr = invalid_address;
        }

        PoolAddressAllocator& operator=(PoolAddressAllocator&& other)
        {
            Base::operator=(std::move(other));
            std::swap(blockCount,other.blockCount);
            std::swap(blockSize,other.blockSize);
			std::swap(freeStack,other.freeStack);
            std::swap(freeStackCtr,other.freeStackCtr);
            return *this;
        }


        inline size_type        alloc_addr( size_type bytes, size_type alignment, size_type hint=0ull) noexcept
        {
            if (freeStackCtr==0u || (blockSize%alignment)!=0u || bytes==0u || bytes>blockSize)
                return invalid_address;

            return freeStack[--freeStackCtr];
        }

        inline void             free_addr(size_type addr, size_type bytes) noexcept
        {
            #ifdef _IRR_DEBUG
                assert(addr>=Base::combinedOffset && (addr-Base::combinedOffset)%blockSize==0 && freeStackCtr<blockCount);
            #endif // _IRR_DEBUG
			freeStack[freeStackCtr++] = addr;
        }

        inline void             reset()
        {
            for (freeStackCtr=0u; freeStackCtr<blockCount; freeStackCtr++)
                freeStack[freeStackCtr] = (blockCount-1u-freeStackCtr)*blockSize+Base::combinedOffset;
        }

        //! conservative estimate, does not account for space lost to alignment
        inline size_type        max_size() const noexcept
        {
            return blockSize;
        }

        //! Most allocators do not support e.g. 1-byte allocations
        inline size_type        min_size() const noexcept
        {
            return blockSize;
        }

        inline size_type        safe_shrink_size(size_type sizeBound, size_type newBuffAlignmentWeCanGuarantee=1u) noexcept
        {
            size_type retval = get_total_size()-Base::alignOffset;
            if (sizeBound>=retval)
                return Base::safe_shrink_size(sizeBound,newBuffAlignmentWeCanGuarantee);

            if (freeStackCtr==0u)
                return Base::safe_shrink_size(retval,newBuffAlignmentWeCanGuarantee);

            auto allocSize = get_allocated_size();
            if (allocSize>sizeBound)
                sizeBound = allocSize;

            auto tmpStackCopy = freeStack+freeStackCtr;

            size_type boundedCount = 0;
            for (size_type i=0; i<freeStackCtr; i++)
            {
                auto freeAddr = freeStack[i];
                if (freeAddr<sizeBound+Base::combinedOffset)
                    continue;

                tmpStackCopy[boundedCount++] = freeAddr;
            }

            if (boundedCount)
            {
                std::make_heap(tmpStackCopy,tmpStackCopy+boundedCount);
                std::sort_heap(tmpStackCopy,tmpStackCopy+boundedCount);
                // could do sophisticated modified version of std::adjacent_find with a binary search, but F'it
                size_type endAddr = (blockCount-1u)*blockSize+Base::combinedOffset;
                size_type i=0u;
                for (;i<boundedCount; i++,endAddr-=blockSize)
                {
                    if (tmpStackCopy[boundedCount-1u-i]!=endAddr)
                        break;
                }

                retval -= i*blockSize;
            }
            return Base::safe_shrink_size(retval,newBuffAlignmentWeCanGuarantee);
        }


        static inline size_type reserved_size(size_type maxAlignment, size_type bufSz, size_type blockSz) noexcept
        {
            size_type maxBlockCount =  bufSz/blockSz;
            return maxBlockCount*sizeof(size_type)*size_type(2u);
        }
        static inline size_type reserved_size(const PoolAddressAllocator<_size_type>& other, size_type bufSz) noexcept
        {
            return reserved_size(other.maxRequestableAlignment,bufSz,other.blockSize);
        }

        inline size_type        get_free_size() const noexcept
        {
            return freeStackCtr*blockSize;
        }
        inline size_type        get_allocated_size() const noexcept
        {
            return (blockCount-freeStackCtr)*blockSize;
        }
        inline size_type        get_total_size() const noexcept
        {
            return blockCount*blockSize+Base::alignOffset;
        }



        inline size_type addressToBlockID(size_type addr) const noexcept
        {
            return (addr-Base::combinedOffset)/blockSize;
        }
    protected:
        size_type   blockCount;
        size_type   blockSize;
        // TODO: free address min-heap and allocated addresses max-heap, packed into the same memory (whatever is not allocated is free)
        // although to implement that one of the heaps would have to be in reverse in the memory (no wiggle room)
        // then can minimize fragmentation due to allocation and give lighting fast returns from `safe_shrink_size`
        // but then should probably have two pool allocators, because doing that changes insertion/removal from O(1) to O(log(N))
		size_type*	freeStack;
        size_type   freeStackCtr;
};


}
}

#include "irr/core/alloc/AddressAllocatorConcurrencyAdaptors.h"

namespace irr
{
namespace core
{

// aliases
template<typename size_type>
using PoolAddressAllocatorST = PoolAddressAllocator<size_type>;

template<typename size_type, class RecursiveLockable>
using PoolAddressAllocatorMT = AddressAllocatorBasicConcurrencyAdaptor<PoolAddressAllocator<size_type>,RecursiveLockable>;

}
}

#endif // __IRR_POOL_ADDRESS_ALLOCATOR_H_INCLUDED__

