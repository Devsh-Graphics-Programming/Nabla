// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_LINEAR_ADDRESS_ALLOCATOR_H_INCLUDED__
#define __NBL_CORE_LINEAR_ADDRESS_ALLOCATOR_H_INCLUDED__

#include "AddressAllocatorBase.h"
#include "nbl/core/alloc/AddressAllocatorBase.h"

namespace nbl
{
namespace core
{


template<typename _size_type>
class LinearAddressAllocator : public AddressAllocatorBase<LinearAddressAllocator<_size_type>,_size_type>
{
        typedef AddressAllocatorBase<LinearAddressAllocator<_size_type>,_size_type> Base;
    public:
        _NBL_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(_size_type);

        #define DUMMY_DEFAULT_CONSTRUCTOR LinearAddressAllocator() : bufferSize(invalid_address), cursor(invalid_address) {}
        GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(DUMMY_DEFAULT_CONSTRUCTOR)
        #undef DUMMY_DEFAULT_CONSTRUCTOR

        virtual ~LinearAddressAllocator() {}

        LinearAddressAllocator(void* reservedSpc, _size_type addressOffsetToApply, _size_type alignOffsetNeeded, _size_type maxAllocatableAlignment, size_type bufSz) noexcept :
                    Base(reservedSpc,addressOffsetToApply,alignOffsetNeeded,maxAllocatableAlignment), bufferSize(bufSz-Base::alignOffset)
        {
            reset();
        }
        
        LinearAddressAllocator(LinearAddressAllocator&& other) noexcept :
            Base(std::move(other))
        {
            std::swap(bufferSize,other.bufferSize);
            std::swap(cursor,other.cursor);
        }

        //! When resizing we require that the copying of data buffer has already been handled by the user of the address allocator
        template<typename... Args>
        LinearAddressAllocator(_size_type newBuffSz, LinearAddressAllocator&& other, Args&&... args) :
            Base(std::move(other),std::forward<Args>(args)...), bufferSize(invalid_address), cursor(invalid_address)
        {
            std::swap(bufferSize,other.bufferSize);
            bufferSize = newBuffSz-Base::alignOffset;
            std::swap(cursor,other.cursor);
        }
        template<typename... Args>
        LinearAddressAllocator(_size_type newBuffSz, const LinearAddressAllocator& other, Args&&... args) :
            Base(other,std::forward<Args>(args)...), bufferSize(invalid_address), cursor(other.cursor)
        {
            bufferSize = newBuffSz-Base::alignOffset;
        }

        LinearAddressAllocator& operator=(LinearAddressAllocator&& other)
        {
            Base::operator=(std::move(other));
            std::swap(bufferSize,other.bufferSize);
            std::swap(cursor,other.cursor);
            return *this;
        }

        //! non-PoT alignments cannot be guaranteed after a resize or move of the backing buffer
        inline size_type    alloc_addr( size_type bytes, size_type alignment, size_type hint=0ull) noexcept
        {
            if (bytes==0 || alignment>Base::maxRequestableAlignment)
                return invalid_address;

            size_type result = core::roundUp(cursor,alignment);
            size_type newCursor = result+bytes;
            if (newCursor>bufferSize || newCursor<cursor) // the extra OR checks for wraparound
                return invalid_address;

            cursor = newCursor;
            return result+Base::combinedOffset;
        }

        // free is a No-OP, only reset can actually reclaim memory
        inline void         free_addr(size_type addr, size_type bytes) noexcept
        {
            return;
        }

        // reset cursor to `c` allocation units
        inline void         reset(size_type c = 0)
        {
            cursor = c;
        }

        //! Conservative estimate, max_size() gives largest size we are sure to be able to allocate
        inline size_type    max_size() const noexcept
        {
            auto worstCursor = roundUp(cursor,Base::maxRequestableAlignment);
            if (worstCursor<bufferSize)
                return bufferSize-worstCursor;
            return 0u;
        }

        //! Most allocators do not support e.g. 1-byte allocations
        inline size_type    min_size() const noexcept
        {
            return 1u;
        }

        inline size_type        safe_shrink_size(size_type sizeBound, size_type newBuffAlignmentWeCanGuarantee=1u) const noexcept
        {
            size_type retval = get_allocated_size();
            return Base::safe_shrink_size(std::max(retval,sizeBound),newBuffAlignmentWeCanGuarantee);
        }

        template<typename... Args>
        static inline size_type reserved_size(const Args&... args) noexcept
        {
            return 0u;
        }

        // total allocatable size, align offset is not allocatable so it's not included
        inline size_type        get_free_size() const noexcept
        {
            return bufferSize-get_allocated_size();
        }
        // the align offset doesn't count as allocated, its just unusable
        inline size_type        get_allocated_size() const noexcept
        {
            return cursor;
        }
        // total size is the metric that includes the align offset
        inline size_type        get_total_size() const noexcept
        {
            return bufferSize+Base::alignOffset;
        }
    protected:
        size_type bufferSize;
        size_type       cursor;
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
using LinearAddressAllocatorST = LinearAddressAllocator<size_type>;

template<typename size_type, class RecursiveLockable>
using LinearAddressAllocatorMT = AddressAllocatorBasicConcurrencyAdaptor<LinearAddressAllocator<size_type>,RecursiveLockable>;

}
}

#endif
