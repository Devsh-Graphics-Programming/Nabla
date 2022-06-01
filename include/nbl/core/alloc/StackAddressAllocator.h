// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_STACK_ADDRESS_ALLOCATOR_H_INCLUDED__
#define __NBL_CORE_STACK_ADDRESS_ALLOCATOR_H_INCLUDED__

#include "BuildConfigOptions.h"

#include "nbl/core/alloc/LinearAddressAllocator.h"


namespace nbl
{
namespace core
{

template<typename _size_type>
class NBL_API StackAddressAllocator  : protected LinearAddressAllocator<_size_type>
{
        typedef LinearAddressAllocator<_size_type>  Base;
    public:
        static constexpr bool supportsArbitraryOrderFrees = false;

        _NBL_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(_size_type);

        #define DUMMY_DEFAULT_CONSTRUCTOR StackAddressAllocator() : minimumAllocSize(invalid_address), allocStackPtr(invalid_address) {}
        GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(DUMMY_DEFAULT_CONSTRUCTOR)
        #undef DUMMY_DEFAULT_CONSTRUCTOR

        virtual ~StackAddressAllocator() {}

        StackAddressAllocator(void* reservedSpc, _size_type addressOffsetToApply, _size_type alignOffsetNeeded, _size_type maxAllocatableAlignment, size_type bufSz, size_type minAllocSize) noexcept :
                    Base(reservedSpc,addressOffsetToApply,alignOffsetNeeded,maxAllocatableAlignment,bufSz), minimumAllocSize(minAllocSize), allocStackPtr(0u) {}

        //! When resizing we require that the copying of data buffer has already been handled by the user of the address allocator
        template<typename... Args>
        StackAddressAllocator(size_type newBuffSz, StackAddressAllocator&& other, Args&&... args) :
                    Base(newBuffSz,std::move(other),std::forward<Args>(args)...),  minimumAllocSize(invalid_address), allocStackPtr(invalid_address)
        {
            std::swap(minimumAllocSize,other.minimumAllocSize);
            std::swap(allocStackPtr,other.allocStackPtr);
        }

        template<typename... Args>
        StackAddressAllocator(size_type newBuffSz, const StackAddressAllocator& other, Args&&... args) :
            Base(newBuffSz, other, std::forward<Args>(args)...),
            minimumAllocSize(other.minimumAllocSize), allocStackPtr(other.allocStackPtr)
        {
        }

        StackAddressAllocator& operator=(StackAddressAllocator&& other)
        {
            static_cast<Base&>(*this) = std::move(other);
            std::swap(minimumAllocSize,other.minimumAllocSize);
            std::swap(allocStackPtr,other.allocStackPtr);
            return *this;
        }

        //! non-PoT alignments cannot be guaranteed after a resize or move of the backing buffer
        inline size_type    alloc_addr( size_type bytes, size_type alignment, size_type hint=0ull) noexcept
        {
            auto oldCursor = Base::cursor;
            size_type result = Base::alloc_addr(std::max(bytes,minimumAllocSize),alignment,hint);
            if (result==invalid_address)
                return invalid_address;

            reinterpret_cast<size_type*>(Base::reservedSpace)[allocStackPtr++] = Base::cursor-oldCursor;
            return result;
        }

        inline void         free_addr(size_type addr, size_type bytes) noexcept
        {
            auto amountToFree = reinterpret_cast<size_type*>(Base::reservedSpace)[--allocStackPtr];
            #ifdef _NBL_DEBUG
                assert(bytes<=amountToFree);
            #endif // _NBL_DEBUG
            Base::cursor -= amountToFree;
            #ifdef _NBL_DEBUG
                assert(Base::cursor+Base::alignOffset<=addr);
            #endif // _NBL_DEBUG
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

        inline size_type    min_size() const noexcept
        {
            return minimumAllocSize;
        }


        static inline size_type reserved_size(size_type maxAlignment, size_type bufSz, size_type minAllocSize) noexcept
        {
            size_type maxAllocCount = bufSz/minAllocSize;
            return maxAllocCount*sizeof(size_type);
        }
        static inline size_type reserved_size(size_type bufSz, const StackAddressAllocator<_size_type>& other) noexcept
        {
            return reserved_size(other.maxRequestableAlignment,bufSz,other.minimumAllocSize);
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

#endif

