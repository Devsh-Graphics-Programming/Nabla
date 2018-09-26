// Copyright (C) 2018 Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW Engine"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_LINEAR_ADDRESS_ALLOCATOR_H_INCLUDED__
#define __IRR_LINEAR_ADDRESS_ALLOCATOR_H_INCLUDED__

#include "IrrCompileConfig.h"

#include "irr/core/alloc/AddressAllocatorBase.h"

namespace irr
{
namespace core
{


template<typename _size_type>
class LinearAddressAllocator : public AddressAllocatorBase<LinearAddressAllocator<_size_type> >
{
        typedef AddressAllocatorBase<LinearAddressAllocator<_size_type> > Base;
    public:
        _IRR_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(_size_type);

        static constexpr bool supportsNullBuffer = true;

        GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(PoolAddressAllocator() : alignOffset(0u), bufferSize(0u) {})

        virtual ~LinearAddressAllocator() {}

        LinearAddressAllocator(void* reservedSpc, void* buffer, size_type buffSz) noexcept : Base(reservedSpc,buffer)
                    alignOffset(calcAlignOffset(buffer,buffSz)), bufferSize(buffSz+alignOffset), cursor(alignOffset) {}

        LinearAddressAllocator(const LinearAddressAllocator& other, void* newReservedSpc, void* newBuffer, size_type newBuffSz) :
                    Base(other,newReservedSpc,newBuffer,newBuffSz), alignOffset(calcAlignOffset(newBuffer,newBuffSz)),
                    bufferSize(newBuffSz+alignOffset), cursor(other.cursor+alignOffset-other.alignOffset)
        {
            if (other.bufferStart&&Base::bufferStart)
            {
#ifdef _DEBUG
                // new pointer must have greater or equal alignment
                assert(findLSB(reinterpret_cast<size_t>(Base::bufferStart)) >= findLSB(reinterpret_cast<size_t>(other.bufferStart)));
#endif // _DEBUG
                memmove(reinterpret_cast<uint8_t*>(Base::bufferStart)+alignOffset,
                        reinterpret_cast<uint8_t*>(other.bufferStart)+other.alignOffset, other.get_allocated_size());
            }
        }

        inline size_type    alloc_addr( size_type bytes, size_type alignment, size_type hint=0ull) noexcept
        {
            if (bytes==0) // || alignment>max_alignment()
                return invalid_address;

            size_type result    = alignUp(cursor,alignment);
            size_type newCursor = result+bytes;
            if (newCursor>bufferSize || newCursor<cursor) //extra or checks for wraparound
                return invalid_address;

            cursor = newCursor;
            return result-alignOffset;
        }

        inline void         free_addr(size_type addr, size_type bytes) noexcept
        {
            return;
        }

        inline void         reset()
        {
            cursor = alignOffset;
        }

        //! conservative estimate, does not account for space lost to alignment
        inline size_type    max_size() const noexcept
        {
            if (cursor>bufferSize)
                return 0u;
            return bufferSize-cursor;
        }

        inline size_type    max_alignment() const noexcept
        {
            size_type tmpSize = bufferSize-alignOffset;
            size_type align = size_type(0x1u)<<findMSB(tmpSize); // what can fit inside the memory?

            while (align)
            {
                size_type tmpStart = alignUp(alignOffset,align); // where would it start if it was aligned to itself
                if (tmpStart+align<=bufferSize) // could it fit?
                    return align;
                align = align>>1u;
            }

            return 1u;
        }

        inline size_type        safe_shrink_size(size_type byteBound=0u) const noexcept
        {
            return get_allocated_size();
        }

        static inline size_type reserved_size(size_type bufSz) noexcept
        {
            return 0u;
        }
        static inline size_type reserved_size(size_type bufSz, const LinearAddressAllocator<_size_type>& other) noexcept
        {
            return 0u;
        }

        inline size_type        get_free_size() const noexcept
        {
            return bufferSize-cursor;
        }
        inline size_type        get_allocated_size() const noexcept
        {
            return cursor-alignOffset;
        }
        inline size_type        get_total_size() const noexcept
        {
            return bufferSize-alignOffset;
        }
    protected:
        const size_type                                     alignOffset;
        const size_type                                     bufferSize;
        size_type                                           cursor;

        static inline size_type calcAlignOffset(void* ptr, size_type bufSz)
        {
            return reinterpret_cast<size_t>(ptr)&((size_type(0x1u)<<findMSB(bufSz))-1u);
        }
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
using LinearAddressAllocatorST = LinearAddressAllocator<size_type>;

template<typename size_type, class BasicLockable>
using LinearAddressAllocatorMT = AddressAllocatorBasicConcurrencyAdaptor<LinearAddressAllocator<size_type>,BasicLockable>;

}
}

#endif // __IRR_LINEAR_ADDRESS_ALLOCATOR_H_INCLUDED__
