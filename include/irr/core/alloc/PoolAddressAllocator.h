// Copyright (C) 2018 Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW Engine"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_POOL_ADDRESS_ALLOCATOR_H_INCLUDED__
#define __IRR_POOL_ADDRESS_ALLOCATOR_H_INCLUDED__

#include "IrrCompileConfig.h"

#include "irr/core/math/irrMath.h"

#include "irr/core/alloc/address_type_traits.h"
#include "irr/core/alloc/AddressAllocatorConcurrencyAdaptors.h"

namespace irr
{
namespace core
{


//! Can only allocate up to a size of a single block, no support for allocations larger than blocksize
template<typename _size_type>
class PoolAddressAllocator
{
    public:
        typedef _size_type                                  size_type;
        typedef typename std::make_signed<size_type>::type  difference_type;
        typedef uint8_t*                                    ubyte_pointer;

        static constexpr size_type                          invalid_address = address_type_traits<size_type>::invalid_address;


        PoolAddressAllocator(void* reservedSpc, size_t alignOff, size_type bufSz, size_type blockSz) noexcept :
                    maxAlignment(size_type(1u)<<findLSB(blockSz)), alignOffset(calcAlignOffset(alignOff,maxAlignment)),
                    blockSize(blockSz), blockCount((bufSz-alignOffset)/blockSz),
                    freeStack(reinterpret_cast<size_type*>(reservedSpc)), freeStackCtr(0u)
        {
            reset();
        }

        inline size_type        alloc_addr( size_type bytes, size_type alignment, size_type hint=0ull) noexcept
        {
            if (freeStackCtr==0u || alignment>maxAlignment || bytes==0u)
                return invalid_address;

            return freeStack[--freeStackCtr];
        }

        inline void             free_addr(size_type addr, size_type bytes) noexcept
        {
#ifdef _DEBUG
            assert(addr>=alignOffset&&freeStackCtr<blockCount);
#endif // _DEBUG
            freeStack[freeStackCtr++] = addr;
        }

        inline void             reset()
        {
            for (freeStackCtr=0u; freeStackCtr<blockCount; freeStackCtr++)
                freeStack[freeStackCtr] = (blockCount-1u-freeStackCtr)*blockSize+alignOffset;
        }

        //! conservative estimate, does not account for space lost to alignment
        inline size_type        max_size() const noexcept
        {
            return blockSize;
        }

        inline size_type        max_alignment() const noexcept
        {
            return maxAlignment;
        }


        template<typename... Args>
        static inline size_type reserved_size(size_t alignOff, size_type bufSz, size_type blockSz, Args&&... args) noexcept
        {
            size_type trueAlign = size_type(1u)<<findLSB(blockSz);
            size_type truncatedOffset = calcAlignOffset(alignOff,trueAlign);
            size_type probBlockCount =  (bufSz-truncatedOffset)/(blockSz+sizeof(size_type))+1u;
            return probBlockCount*sizeof(size_type);
        }
    protected:
        const size_type                                     maxAlignment;
        const size_type                                     alignOffset;
        const size_type                                     blockSize;
        const size_type                                     blockCount;

        size_type* const                                    freeStack;
        size_type                                           freeStackCtr;
    private:
        static inline size_type calcAlignOffset(size_t o, size_type a)
        {
            size_type remainder = o&(a-1u);
            return (a-remainder)&(a-1u);
        }
};

//! Ideas for general pool allocator (with support for arrays)
// sorted free vector - binary search insertion on free, linear search on allocate
// sorted free ranges - binary search on free, possible insertion or deletion (less stuff to search)

// how to find the correct free range size?
// sorted per-size vector of ranges (2x lower overhead)


// aliases
template<typename size_type>
using PoolAddressAllocatorST = PoolAddressAllocator<size_type>;

template<typename size_type, class BasicLockable>
using PoolAddressAllocatorMT = AddressAllocatorBasicConcurrencyAdaptor<PoolAddressAllocator<size_type>,BasicLockable>;

}
}

#endif // __IRR_POOL_ADDRESS_ALLOCATOR_H_INCLUDED__

