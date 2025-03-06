// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_POOL_ADDRESS_ALLOCATOR_H_INCLUDED__
#define __NBL_CORE_POOL_ADDRESS_ALLOCATOR_H_INCLUDED__

#include "BuildConfigOptions.h"

#include "nbl/core/alloc/AddressAllocatorBase.h"

namespace nbl
{
namespace core
{


//! Can only allocate up to a size of a single block, no support for allocations larger than blocksize
template<typename _size_type>
class PoolAddressAllocator : public AddressAllocatorBase<PoolAddressAllocator<_size_type>,_size_type>
{
    private:
        using base_t = AddressAllocatorBase<PoolAddressAllocator<_size_type>,_size_type>;

        void copyState(const PoolAddressAllocator& other, _size_type newBuffSz)
        {
            if (blockCount>other.blockCount)
                freeStackCtr = blockCount-other.blockCount;

            #ifdef _NBL_DEBUG
                assert(base_t::checkResize(newBuffSz,base_t::alignOffset));
            #endif // _NBL_DEBUG

            for (_size_type i=0u; i<freeStackCtr; i++)
                getFreeStack(i) = (blockCount-1u-i)*blockSize+base_t::combinedOffset;

            for (_size_type i=0; i<other.freeStackCtr; i++)
            {
                _size_type freeEntry = other.getFreeStack(i)-other.base_t::combinedOffset;
                // check in case of shrink
                if (freeEntry<blockCount*blockSize)
                    getFreeStack(freeStackCtr++) = freeEntry+base_t::combinedOffset;
            }
        }

        inline bool safe_shrink_size_common(_size_type& sizeBound, _size_type newBuffAlignmentWeCanGuarantee) noexcept
        {
            _size_type capacity = get_total_size()-base_t::alignOffset;
            if (sizeBound>=capacity)
                return false;

            if (freeStackCtr==0u)
            {
                sizeBound = capacity;
                return false;
            }

            auto allocSize = get_allocated_size();
            if (allocSize>sizeBound)
                sizeBound = allocSize;
            return true;
        }

    public:
        _NBL_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(_size_type);

        static constexpr bool supportsNullBuffer = true;

        PoolAddressAllocator() : blockSize(1u), blockCount(0u) {}

        virtual ~PoolAddressAllocator() {}

        PoolAddressAllocator(void* reservedSpc, _size_type addressOffsetToApply, _size_type alignOffsetNeeded, _size_type maxAllocatableAlignment, size_type bufSz, size_type blockSz) noexcept :
					base_t(reservedSpc,addressOffsetToApply,alignOffsetNeeded,maxAllocatableAlignment),
						blockCount((bufSz-alignOffsetNeeded)/blockSz), blockSize(blockSz), freeStackCtr(0u)
        {
            reset();
        }

        //! When resizing we require that the copying of data buffer has already been handled by the user of the address allocator
        template<typename... Args>
        PoolAddressAllocator(_size_type newBuffSz, PoolAddressAllocator&& other, Args&&... args) noexcept :
					base_t(other,std::forward<Args>(args)...),
						blockCount((newBuffSz-base_t::alignOffset)/other.blockSize), blockSize(other.blockSize), freeStackCtr(0u)
        {
            copyState(other, newBuffSz);

            other.invalidate();
        }
        template<typename... Args>
        PoolAddressAllocator(_size_type newBuffSz, const PoolAddressAllocator& other, Args&&... args) noexcept :
            base_t(other, std::forward<Args>(args)...),
            blockCount((newBuffSz-base_t::alignOffset)/other.blockSize), blockSize(other.blockSize), freeStackCtr(0u)
        {
            copyState(other, newBuffSz);
        }

        PoolAddressAllocator& operator=(PoolAddressAllocator&& other)
        {
            base_t::operator=(std::move(other));
            blockCount = other.blockCount;
            blockSize = other.blockSize;
            freeStackCtr = other.freeStackCtr;
            other.invalidateLocal();
            return *this;
        }


        inline size_type        alloc_addr( size_type bytes, size_type alignment, size_type hint=0ull) noexcept
        {
            if (freeStackCtr==0u || (blockSize%alignment)!=0u || bytes==0u || bytes>blockSize)
                return invalid_address;

            return getFreeStack(--freeStackCtr);
        }

        inline void             free_addr(size_type addr, size_type bytes) noexcept
        {
            #ifdef _NBL_DEBUG
                assert(addr>=base_t::combinedOffset && (addr-base_t::combinedOffset)%blockSize==0 && freeStackCtr<blockCount);
            #endif // _NBL_DEBUG
			getFreeStack(freeStackCtr++) = addr;
        }

        inline void             reset()
        {
            for (freeStackCtr=0u; freeStackCtr<blockCount; freeStackCtr++)
                getFreeStack(freeStackCtr) = (blockCount-1u-freeStackCtr)*blockSize+base_t::combinedOffset;
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
            if (safe_shrink_size_common(sizeBound,newBuffAlignmentWeCanGuarantee))
            {
                auto tmpStackCopy = &getFreeStack(freeStackCtr);

                size_type boundedCount = 0;
                for (size_type i=0; i<freeStackCtr; i++)
                {
                    auto freeAddr = getFreeStack(i);
                    if (freeAddr<sizeBound+base_t::combinedOffset)
                        continue;

                    tmpStackCopy[boundedCount++] = freeAddr;
                }

                if (boundedCount)
                {
                    std::make_heap(tmpStackCopy,tmpStackCopy+boundedCount);
                    std::sort_heap(tmpStackCopy,tmpStackCopy+boundedCount);
                    // could do sophisticated modified version of std::adjacent_find with a binary search, but F'it
                    size_type endAddr = (blockCount-1u)*blockSize+base_t::combinedOffset;
                    size_type i=0u;
                    for (;i<boundedCount; i++,endAddr-=blockSize)
                    {
                        if (tmpStackCopy[boundedCount-1u-i]!=endAddr)
                            break;
                    }

                    sizeBound -= i*blockSize;
                }
            }
            return base_t::safe_shrink_size(sizeBound,newBuffAlignmentWeCanGuarantee);
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
            return blockCount*blockSize+base_t::alignOffset;
        }



        inline size_type addressToBlockID(size_type addr) const noexcept
        {
            return (addr-base_t::combinedOffset)/blockSize;
        }
    protected:

        /**
        * @brief Invalidates only fields from this class extension
        */
        void invalidateLocal()
        {
            blockCount = invalid_address;
            blockSize = invalid_address;
            freeStackCtr = invalid_address;
        }

        /**
        * @brief Invalidates all fields
        */
        void invalidate()
        {
            base_t::invalidate();
            invalidateLocal();
        }

        size_type   blockCount;
        size_type   blockSize;
        // TODO: free address min-heap and allocated addresses max-heap, packed into the same memory (whatever is not allocated is free)
        // although to implement that one of the heaps would have to be in reverse in the memory (no wiggle room)
        // then can minimize fragmentation due to allocation and give lighting fast returns from `safe_shrink_size`
        // but then should probably have two pool allocators, because doing that changes insertion/removal from O(1) to O(log(N))
        size_type   freeStackCtr;

        inline size_type& getFreeStack(size_type i) {return reinterpret_cast<size_type*>(base_t::reservedSpace)[i];}
        inline const size_type& getFreeStack(size_type i) const {return reinterpret_cast<const size_type*>(base_t::reservedSpace)[i];} 
};


}
}

#include "nbl/core/alloc/AddressAllocatorConcurrencyAdaptors.h"

namespace nbl
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

#endif

