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

        PoolAddressAllocator(void* reservedSpc, void* buffer, size_type maxAllocatableAlignment, size_type bufSz, size_type blockSz) noexcept :
                    Base(reservedSpc,buffer,std::min(size_type(1u)<<size_type(findLSB(blockSz)),maxAllocatableAlignment)), blockCount((bufSz-Base::alignOffset)/blockSz), blockSize(blockSz), freeStackCtr(0u)
        {
            reset();
        }

        //! When resizing we require that the copying of data buffer has already been handled by the user of the address allocator even if `supportsNullBuffer==true`
        PoolAddressAllocator(const PoolAddressAllocator& other, void* newReservedSpc, void* newBuffer, size_type newBuffSz) noexcept :
                    Base(other,newReservedSpc,newBuffer,newBuffSz), blockCount((newBuffSz-Base::alignOffset)/other.blockSize), blockSize(other.blockSize),
                    freeStackCtr(blockCount>other.blockCount ? (blockCount-other.blockCount):0u)
        {
            auto freeStack = reinterpret_cast<size_type*>(Base::reservedSpace);
            for (size_type i=0u; i<freeStackCtr; i++)
                freeStack[i] = (blockCount-1u-i)*blockSize+Base::alignOffset;

            for (size_type i=0; i<other.freeStackCtr; i++)
            {
                size_type freeEntry = other.addressToBlockID(reinterpret_cast<size_type*>(other.reservedSpace)[i]);

                if (freeEntry<blockCount)
                    freeStack[freeStackCtr++] = freeEntry*blockSize+Base::alignOffset;
            }
        }

        PoolAddressAllocator& operator=(PoolAddressAllocator&& other)
        {
            Base::operator=(std::move(other));
            blockCount = other.blockCount;
            blockSize = other.blockSize;
            freeStackCtr = other.freeStackCtr;
            return *this;
        }


        inline size_type        alloc_addr( size_type bytes, size_type alignment, size_type hint=0ull) noexcept
        {
            if (freeStackCtr==0u || (blockSize%alignment)==0u || bytes==0u || bytes>blockSize)
                return invalid_address;

            auto freeStack = reinterpret_cast<size_type*>(Base::reservedSpace);
            return freeStack[--freeStackCtr];
        }

        inline void             free_addr(size_type addr, size_type bytes) noexcept
        {
#ifdef _DEBUG
            assert(addr>=Base::alignOffset && (addr-Base::alignOffset)%blockSize==0 && freeStackCtr<blockCount);
#endif // _DEBUG
            auto freeStack = reinterpret_cast<size_type*>(Base::reservedSpace);
            freeStack[freeStackCtr++] = addr;
        }

        inline void             reset()
        {
            auto freeStack = reinterpret_cast<size_type*>(Base::reservedSpace);
            for (freeStackCtr=0u; freeStackCtr<blockCount; freeStackCtr++)
                freeStack[freeStackCtr] = (blockCount-1u-freeStackCtr)*blockSize+Base::alignOffset;
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

        inline size_type        safe_shrink_size(size_type byteBound=0u, size_type newBuffAlignmentWeCanGuarantee=1u) const noexcept
        {
            size_type retval = get_total_size()-Base::alignOffset;
            if (byteBound>=retval)
                return Base::safe_shrink_size(byteBound,newBuffAlignmentWeCanGuarantee);

            if (freeStackCtr==0u)
                return Base::safe_shrink_size(retval,newBuffAlignmentWeCanGuarantee);

            auto allocSize = get_allocated_size();
            if (allocSize>byteBound)
                byteBound = allocSize;

            auto freeStack = reinterpret_cast<size_type*>(Base::reservedSpace);
            size_type* tmpStackCopy = reinterpret_cast<size_type*>(_IRR_ALIGNED_MALLOC(freeStackCtr*sizeof(size_type),_IRR_SIMD_ALIGNMENT));

            size_type boundedCount = 0;
            for (size_type i=0; i<freeStackCtr; i++)
            {
                auto freeAddr = freeStack[i];
                if (freeAddr<byteBound+Base::alignOffset)
                    continue;

                tmpStackCopy[boundedCount++] = freeAddr;
            }

            if (boundedCount)
            {
                std::make_heap(tmpStackCopy,tmpStackCopy+boundedCount);
                std::sort_heap(tmpStackCopy,tmpStackCopy+boundedCount);
                // could do sophisticated modified binary search, but f'it
                size_type endAddr = (blockCount-1u)*blockSize+Base::alignOffset;
                size_type i=0u;
                for (;i<boundedCount; i++,endAddr-=blockSize)
                {
                    if (tmpStackCopy[boundedCount-1u-i]!=endAddr)
                        break;
                }

                retval -= i*blockSize;
            }
            _IRR_ALIGNED_FREE(tmpStackCopy);
            return Base::safe_shrink_size(retval,newBuffAlignmentWeCanGuarantee);
        }


        static inline size_type reserved_size(size_type bufSz, size_type maxAlignment, size_type blockSz) noexcept
        {
            size_type maxBlockCount =  bufSz/blockSz;
            return maxBlockCount*sizeof(size_type);
        }
        static inline size_type reserved_size(size_type bufSz, const PoolAddressAllocator<_size_type>& other) noexcept
        {
            return reserved_size(bufSz,other.maxRequestableAlignment,other.blockSize);
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
            return (addr-Base::alignOffset)/blockSize;
        }
    protected:
        size_type   blockCount;
        size_type   blockSize;

        size_type   freeStackCtr;
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

