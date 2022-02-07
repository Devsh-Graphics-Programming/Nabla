// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_ITERATABLE_POOL_ADDRESS_ALLOCATOR_H_INCLUDED__
#define __NBL_CORE_ITERATABLE_POOL_ADDRESS_ALLOCATOR_H_INCLUDED__

#include <algorithm>

#include "nbl/core/alloc/PoolAddressAllocator.h"

namespace nbl
{
namespace core
{
//! Can only allocate up to a size of a single block, no support for allocations larger than blocksize
template<typename _size_type>
class IteratablePoolAddressAllocator : protected PoolAddressAllocator<_size_type>
{
protected:
    inline size_type* begin() { return &getFreeStack(Base::freeStackCtr); }
    inline size_type& getIteratorOffset(size_type i) { return reinterpret_cast<size_type*>(Base::reservedSpace)[Base::blockCount + i]; }
    inline const size_type& getIteratorOffset(size_type i) const { return reinterpret_cast<const size_type*>(Base::reservedSpace)[Base::blockCount + i]; }

private:
    using Base = PoolAddressAllocator<_size_type>;

    void copySupplementaryState(const IteratablePoolAddressAllocator& other, _size_type newBuffSz)
    {
        std::copy(other.begin(), other.end(), begin());
        for(auto i = 0u; i < std::min(blockCount, other.blockCount); i++)
            getIteratorOffset(i) = other.getIteratorOffset(i);
    }
    // use [freeStackCtr,blockCount) as the iteratable range
    // use [blockCount,blockCount*2u) to store backreferences to iterators
public:
    _NBL_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(_size_type);

    IteratablePoolAddressAllocator()
        : Base() {}
    virtual ~IteratablePoolAddressAllocator() {}

    IteratablePoolAddressAllocator(void* reservedSpc, _size_type addressOffsetToApply, _size_type alignOffsetNeeded, _size_type maxAllocatableAlignment, size_type bufSz, size_type blockSz) noexcept
        : Base(reservedSpc, addressOffsetToApply, alignOffsetNeeded, maxAllocatableAlignment, bufSz, blockSz) {}

    //! When resizing we require that the copying of data buffer has already been handled by the user of the address allocator
    template<typename... Args>
    IteratablePoolAddressAllocator(_size_type newBuffSz, IteratablePoolAddressAllocator&& other, Args&&... args) noexcept
        : Base(newBuffSz, std::move(other), std::forward<Args>(args)...)
    {
        copyState
    }

    template<typename... Args>
        IteratablePoolAddressAllocator(_size_type newBuffSz, const IteratablePoolAddressAllocator& other, Args&&... args) noexcept
        : Base(newBuffSz, other, std::forward<Args>(args)...){
              copyState}

          IteratablePoolAddressAllocator
          & operator=(IteratablePoolAddressAllocator&& other)
    {
        Base::operator=(std::move(other));
        return *this;
    }

    //! Functions that actually differ
    inline size_type alloc_addr(size_type bytes, size_type alignment, size_type hint = 0ull) noexcept
    {
        const size_type allocatedAddress = Base::alloc_addr(bytes, alignment, hint);
        if(allocatedAddress != invalid_address)
        {
            *begin() = allocatedAddress;
            getIteratorOffset(addressToBlockID(allocatedAddress)) = freeStackCtr;
        }
        return allocatedAddress;
    }

    inline void free_addr(size_type addr, size_type bytes) noexcept
    {
        const size_type iteratorOffset = getIteratorOffset(addressToBlockID(addr));
#ifdef _NBL_DEBUG
        assert(iteratorOffset >= freeStackCtr);
#endif
        // swap the erased element with either end of the array in the contiguous array
        // not using a swap cause it doesn't matter where the erased element points
        const size_type otherNodeOffset = *begin();
        reinterpret_cast<size_type*>(Base::reservedSpace)[iteratorOffset] = otherNodeOffset;
        // but I need to patch up the back-link of the moved element
        getIteratorOffset(addressToBlockID(otherNodeOffset)) = iteratorOffset;

        Base::free_addr(addr, bytes);
    }

    // gets a range of all the allocated addresses
    inline const size_type* begin() const { return &getFreeStack(Base::freeStackCtr); }
    inline const size_type* end() const { return &getFreeStack(Base::blockCount); }

    inline size_type safe_shrink_size(size_type sizeBound, size_type newBuffAlignmentWeCanGuarantee = 1u) noexcept
    {
        if(safe_shrink_size_common(sizeBound, newBuffAlignmentWeCanGuarantee))
        {
            assert(begin() != end());  // we already checked that freeStackCtr>0
            sizeBound = *std::max_element(begin(), end());
        }
        return AddressAllocatorBase<PoolAddressAllocator<_size_type>, _size_type>::safe_shrink_size(sizeBound, newBuffAlignmentWeCanGuarantee);
    }

    static inline size_type reserved_size(size_type maxAlignment, size_type bufSz, size_type blockSz) noexcept
    {
        size_type maxBlockCount = bufSz / blockSz;
        return maxBlockCount * sizeof(size_type) * size_type(2u);
    }
    static inline size_type reserved_size(const IteratablePoolAddressAllocator<_size_type>& other, size_type bufSz) noexcept
    {
        return reserved_size(other.maxRequestableAlignment, bufSz, other.blockSize);
    }

    inline void reset()
    {
        Base::reset();
    }
    inline size_type max_size() const noexcept
    {
        return Base::max_size();
    }
    inline size_type min_size() const noexcept
    {
        return Base::min_size();
    }
    inline size_type get_free_size() const noexcept
    {
        return Base::get_free_size();
    }
    inline size_type get_allocated_size() const noexcept
    {
        return Base::get_allocated_size();
    }
    inline size_type get_total_size() const noexcept
    {
        return Base::get_total_size();
    }
    inline size_type addressToBlockID(size_type addr) const noexcept
    {
        return Base::addressToBlockID(addr);
    }
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
using IteratablePoolAddressAllocatorST = IteratablePoolAddressAllocator<size_type>;

template<typename size_type, class RecursiveLockable>
using IteratablePoolAddressAllocatorMT = AddressAllocatorBasicConcurrencyAdaptor<IteratablePoolAddressAllocator<size_type>, RecursiveLockable>;

}
}

#endif
