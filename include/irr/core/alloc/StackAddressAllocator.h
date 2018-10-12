// Copyright (C) 2018 Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW Engine"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_STACK_ADDRESS_ALLOCATOR_H_INCLUDED__
#define __IRR_STACK_ADDRESS_ALLOCATOR_H_INCLUDED__

#include "IrrCompileConfig.h"

#include "irr/core/alloc/AddressAllocatorBase.h"


namespace irr
{
namespace core
{

template<typename _size_type>
class StackAddressAllocator  : protected LinearAddressAllocator<_size_type>
{
        typedef LinearAddressAllocator<_size_type>  Base;
    public:
        static constexpr bool supportsArbitraryOrderFrees = false;

        _IRR_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(_size_type);

        static constexpr bool supportsNullBuffer = true;

        #define DUMMY_DEFAULT_CONSTRUCTOR StackAddressAllocator() : minimumAllocSize(0u), allocStackPtr(0u) {}
        GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(DUMMY_DEFAULT_CONSTRUCTOR)
        #undef DUMMY_DEFAULT_CONSTRUCTOR

        virtual ~StackAddressAllocator() {}

        StackAddressAllocator(void* reservedSpc, void* buffer, size_type maxAllocatableAlignment, size_type buffSz, size_type minAllocSize) noexcept :
                    Base(reservedSpc,buffer,maxAllocatableAlignment,buffSz), minimumAllocSize(minAllocSize), allocStackPtr(0u) {}

        StackAddressAllocator(const StackAddressAllocator& other, void* newReservedSpc, void* newBuffer, size_type newBuffSz) :
                    Base(other,newReservedSpc,newBuffer,newBuffSz), minimumAllocSize(other.minimumAllocSize), allocStackPtr(other.allocStackPtr)
        {
        }

        StackAddressAllocator& operator=(StackAddressAllocator&& other)
        {
            static_cast<Base&>(*this) = std::move(other);
            minimumAllocSize = other.minimumAllocSize;
            allocStackPtr = other.allocStackPtr;
            return *this;
        }

        inline size_type    alloc_addr( size_type bytes, size_type alignment, size_type hint=0ull) noexcept
        {
            if (bytes==0 || alignment>max_alignment())
                return invalid_address;

            size_type result    = alignUp(cursor,alignment);
            size_type newCursor = result+std::max(bytes,minAllocSize);
            if (newCursor>bufferSize || newCursor<cursor) //extra OR checks for wraparound
                return invalid_address;

            reinterpret_cast<size_type*>(Base::reservedSpace)[allocStackPtr++] = newCursor-cursor;

            cursor = newCursor;
            return result+Base::alignOffset;
        }

        inline void         free_addr(size_type addr, size_type bytes) noexcept
        {
            auto amountToFree = reinterpret_cast<size_type*>(Base::reservedSpace)[--allocStackPtr];
#ifdef _DEBUG
            assert(bytes<=amountToFree);
#endif // _DEBUG
            cursor -= amountToFree;
#ifdef _DEBUG
            assert(cursor+Base::alignOffset<=addr);
#endif // _DEBUG
        }

        inline void         reset()
        {
            Base::reset();
            allocStackPtr = 0u;
        }

        inline size_type    max_size() const noexcept
        {
            return Base::max_size();
        }


        static inline size_type reserved_size(size_type bufSz, size_type maxAlignment, size_type minAllocSize) noexcept
        {
            size_type maxAllocCount = bufSz/minAllocSize;
            return maxAllocCount*sizeof(size_type);
        }
        static inline size_type reserved_size(size_type bufSz, const PoolAddressAllocator<_size_type>& other) noexcept
        {
            return reserved_size(bufSz,other.maxRequestableAlignment,other.minimumAllocSize);
        }


        inline size_type        get_free_size() const noexcept
        {
            return Base::get_free_size();
        }
        inline size_type        get_allocated_size() const noexcept
        {
            return Base::get_allocated_size();
        }
        inline size_type        get_total_size() const noexcept
        {
            return Base::get_total_size();
        }
    protected:
        size_type minimumAllocSize;
        size_type allocStackPtr;
};


// aliases no point for a Multithread Stack allocator
template<typename size_type>
using StackAddressAllocatorST = StackAddressAllocator<size_type>;

}
}

#endif // __IRR_STACK_ADDRESS_ALLOCATOR_H_INCLUDED__

