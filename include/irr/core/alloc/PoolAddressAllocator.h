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
class PoolAddressAllocator : public AddressAllocatorBase<PoolAddressAllocator<_size_type> >
{
    private:
        typedef AddressAllocatorBase<PoolAddressAllocator<_size_type> > Base;
    public:
        _IRR_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(_size_type);

        static constexpr bool supportsNullBuffer = true;

        #define DUMMY_DEFAULT_CONSTRUCTOR PoolAddressAllocator() : maxAlignment(0x40000000u), alignOffset(0u), blockSize(1u), blockCount(0u) {}
        GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(DUMMY_DEFAULT_CONSTRUCTOR)
        #undef DUMMY_DEFAULT_CONSTRUCTOR

        virtual ~PoolAddressAllocator() {}

        PoolAddressAllocator(void* reservedSpc, void* buffer, size_type bufSz, size_type blockSz) noexcept :
                    AddressAllocatorBase<PoolAddressAllocator<_size_type> >(reservedSpc,buffer), maxAlignment(size_type(1u)<<findLSB(blockSz)),
                    alignOffset(calcAlignOffset(reinterpret_cast<size_t>(buffer),maxAlignment)), blockCount((bufSz-alignOffset)/blockSz),
                    blockSize(blockSz), freeStackCtr(0u)
        {
            reset();
        }

        PoolAddressAllocator(const PoolAddressAllocator& other, void* newReservedSpc, void* newBuffer, size_type newBuffSz) noexcept :
                    AddressAllocatorBase<PoolAddressAllocator<_size_type> >(newReservedSpc,newBuffer), maxAlignment(other.maxAlignment),
                    alignOffset(calcAlignOffset(reinterpret_cast<size_t>(newBuffer),maxAlignment)),
                    blockCount((newBuffSz-alignOffset)/other.blockSize), blockSize(other.blockSize), freeStackCtr(0u)
        {
            for (size_type i=0; i<other.freeStackCtr; i++)
            {
                size_type freeEntry = reinterpret_cast<size_type*>(other.reservedSpace)[i];

                auto freeStack = reinterpret_cast<size_type*>(Base::reservedSpace);
                if (freeEntry<blockCount)
                    freeStack[freeStackCtr++] = freeEntry;
            }
        }

        inline size_type        alloc_addr( size_type bytes, size_type alignment, size_type hint=0ull) noexcept
        {
            if (freeStackCtr==0u || alignment>maxAlignment || bytes==0u)
                return invalid_address;

            auto freeStack = reinterpret_cast<size_type*>(Base::reservedSpace);
            return freeStack[--freeStackCtr];
        }

        inline void             free_addr(size_type addr, size_type bytes) noexcept
        {
#ifdef _DEBUG
            assert(addr<alignOffset&&freeStackCtr<blockCount);
#endif // _DEBUG
            auto freeStack = reinterpret_cast<size_type*>(Base::reservedSpace);
            freeStack[freeStackCtr++] = addr;
        }

        inline void             reset()
        {
            auto freeStack = reinterpret_cast<size_type*>(Base::reservedSpace);
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

        inline size_type        safe_shrink_size(size_type bound=0u) const noexcept
        {
            size_type retval = (blockCount+1u)*blockSize;
            if (freeStackCtr==0u)
                return retval;

            auto allocSize = (blockCount-freeStackCtr)*blockSize;
            if (allocSize>bound)
                bound = allocSize;

            auto freeStack = reinterpret_cast<size_type*>(Base::reservedSpace);
            size_type* tmpStackCopy = _IRR_ALIGNED_MALLOC(freeStackCtr*sizeof(size_type),_IRR_SIMD_ALIGNMENT);

            size_type boundedCount = 0;
            for (size_type i=0; i<freeStackCtr; i++)
            {
                if (freeStack[i]<bound+alignOffset)
                    continue;

                tmpStackCopy[boundedCount++] = freeStack[i];
            }

            std::make_heap(tmpStackCopy,tmpStackCopy+boundedCount);
            std::sort_heap(tmpStackCopy,tmpStackCopy+boundedCount);
            // could do sophisticated modified binary search, but f'it
            size_type i=1u;
            for (size_type firstWrong = boundedCount-1; i>=boundedCount; firstWrong--,i++)
            {
                if (tmpStackCopy[firstWrong]!=blockCount-i)
                    break;
            }

            retval -= (i-1u)*blockSize;

            _IRR_ALIGNED_FREE(tmpStackCopy);
            return retval;
        }


        template<typename... Args>
        static inline size_type reserved_size(size_type bufSz, size_type blockSz, const Args&... args) noexcept
        {
            size_type trueAlign = size_type(1u)<<findLSB(blockSz);
            size_type truncatedOffset = calcAlignOffset(0x8000000000000000ull-size_t(_IRR_SIMD_ALIGNMENT),trueAlign);
            size_type probBlockCount =  (bufSz-truncatedOffset)/blockSz;
            return probBlockCount*sizeof(size_type);
        }
    protected:
        const size_type                                     maxAlignment;
        const size_type                                     alignOffset;
        const size_type                                     blockCount;
        const size_type                                     blockSize;

        size_type                                           freeStackCtr;

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

template<typename size_type, class BasicLockable>
using PoolAddressAllocatorMT = AddressAllocatorBasicConcurrencyAdaptor<PoolAddressAllocator<size_type>,BasicLockable>;

}
}

#endif // __IRR_POOL_ADDRESS_ALLOCATOR_H_INCLUDED__

