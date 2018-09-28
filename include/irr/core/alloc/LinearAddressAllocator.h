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
class LinearAddressAllocator : public AddressAllocatorBase<LinearAddressAllocator<_size_type>,_size_type>
{
        typedef AddressAllocatorBase<LinearAddressAllocator<_size_type>,_size_type> Base;
    public:
        _IRR_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(_size_type);

        /// C++17 inline static constexpr bool supportsNullBuffer = true;

        #define DUMMY_DEFAULT_CONSTRUCTOR LinearAddressAllocator() : bufferSize(0u), cursor(0u) {}
        GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(DUMMY_DEFAULT_CONSTRUCTOR)
        #undef DUMMY_DEFAULT_CONSTRUCTOR

        virtual ~LinearAddressAllocator() {}

        LinearAddressAllocator(void* reservedSpc, void* buffer, size_type maxAllocatableAlignment, size_type buffSz) noexcept :
                    Base(reservedSpc,buffer,maxAllocatableAlignment), bufferSize(buffSz-Base::alignOffset), cursor(0u) {}

        LinearAddressAllocator(const LinearAddressAllocator& other, void* newReservedSpc, void* newBuffer, size_type newBuffSz) :
                    Base(other,newReservedSpc,newBuffer,newBuffSz), bufferSize(newBuffSz-Base::alignOffset), cursor(other.cursor)
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

        LinearAddressAllocator& operator=(LinearAddressAllocator&& other)
        {
            static_cast<Base&>(*this) = std::move(other);
            bufferSize = other.bufferSize;
            cursor = other.cursor;
            return *this;
        }

        inline size_type    alloc_addr( size_type bytes, size_type alignment, size_type hint=0ull) noexcept
        {
            if (bytes==0 || alignment>max_alignment())
                return invalid_address;

            size_type result    = alignUp(cursor,alignment);
            size_type newCursor = result+bytes;
            if (newCursor>bufferSize || newCursor<cursor) //extra OR checks for wraparound
                return invalid_address;

            cursor = newCursor;
            return result+Base::alignOffset;
        }

        inline void         free_addr(size_type addr, size_type bytes) noexcept
        {
            return;
        }

        inline void         reset()
        {
            cursor = 0u;
        }

        //! conservative estimate, does not account for space lost to alignment
        inline size_type    max_size() const noexcept
        {
            if (cursor>bufferSize)
                return 0u;
            return bufferSize-cursor;
        }

        inline size_type        safe_shrink_size(size_type byteBound=0u, size_type newBuffAlignmentWeCanGuarantee=1u) const noexcept
        {
            size_type retval = get_allocated_size();
            if (newBuffAlignmentWeCanGuarantee<Base::maxRequestableAlignment)
                retval += Base::maxRequestableAlignment-newBuffAlignmentWeCanGuarantee;
            return retval;
        }

        template<typename... Args>
        static inline size_type reserved_size(const Args&... args) noexcept
        {
            return 0u;
        }

        inline size_type        get_free_size() const noexcept
        {
            return bufferSize-cursor;
        }
        inline size_type        get_allocated_size() const noexcept
        {
            return cursor;
        }
        inline size_type        get_total_size() const noexcept
        {
            return bufferSize;
        }
    protected:
        const size_type bufferSize;
        size_type       cursor;
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
