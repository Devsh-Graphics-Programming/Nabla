// Copyright (C) 2018 Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW Engine"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_LINEAR_ADDRESS_ALLOCATOR_H_INCLUDED__
#define __IRR_LINEAR_ADDRESS_ALLOCATOR_H_INCLUDED__

#include "IrrCompileConfig.h"

#include "irr/core/alloc/address_type_traits.h"
#include "irr/core/alloc/AddressAllocatorConcurrencyAdaptors.h"

namespace irr
{
namespace core
{


template<typename _size_type>
class LinearAddressAllocator
{
    public:
        typedef _size_type                                  size_type;
        typedef typename std::make_signed<size_type>::type  difference_type;
        typedef uint8_t*                                    ubyte_pointer;

        static constexpr size_type                          invalid_address = address_type_traits<size_type>::invalid_address;
        static constexpr size_type                          max_alignment   = size_type(1)<<size_type(sizeof(size_type)*8-1);


        LinearAddressAllocator(size_type alignOff, size_type bufSz) noexcept : alignOffset(alignOff),
                    bufferSize(bufSz+alignOff), cursor(alignOff) {}

        inline size_type    alloc_addr( size_type bytes, size_type alignment, size_type hint=0ull) noexcept
        {
            if (bytes==0)
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
        inline size_type    max_size() noexcept
        {
            if (cursor>bufferSize)
                return 0u;
            return bufferSize-cursor;
        }

    protected:
        virtual ~LinearAddressAllocator() {}

        const size_type                                     alignOffset;
        const size_type                                     bufferSize;
        size_type                                           cursor;
};


// aliases
template<typename size_type>
using LinearAddressAllocatorST = LinearAddressAllocator<size_type>;

template<typename size_type, class BasicLockable>
using LinearAddressAllocatorMT = AddressAllocatorBasicConcurrencyAdaptor<LinearAddressAllocator<size_type>,BasicLockable>;

}
}

#endif // __IRR_LINEAR_ADDRESS_ALLOCATOR_H_INCLUDED__
