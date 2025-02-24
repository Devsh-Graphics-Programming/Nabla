// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_ITERATABLE_POOL_ADDRESS_ALLOCATOR_H_INCLUDED__
#define __NBL_CORE_ITERATABLE_POOL_ADDRESS_ALLOCATOR_H_INCLUDED__


#include<algorithm>

#include "nbl/core/alloc/PoolAddressAllocator.h"


namespace nbl
{
namespace core
{


//! Can only allocate up to a size of a single block, no support for allocations larger than blocksize
template<typename _size_type>
class IteratablePoolAddressAllocator : protected PoolAddressAllocator<_size_type>
{
        using base_t = PoolAddressAllocator<_size_type>;
    protected:
        inline _size_type* begin() { return &base_t::getFreeStack(base_t::freeStackCtr); }
        inline _size_type& getIteratorOffset(_size_type i) {return reinterpret_cast<_size_type*>(base_t::reservedSpace)[base_t::blockCount+i];}
        inline const _size_type& getIteratorOffset(_size_type i) const {return reinterpret_cast<const _size_type*>(base_t::reservedSpace)[base_t::blockCount+i];}

    private:

        void copySupplementaryState(const IteratablePoolAddressAllocator& other, _size_type newBuffSz)
        {
            std::copy(other.begin(),other.end(),begin());
            for (auto i=0u; i<std::min(base_t::blockCount,other.blockCount); i++)
                getIteratorOffset(i) = other.getIteratorOffset(i);
        }
        // use [freeStackCtr,blockCount) as the iteratable range
        // use [blockCount,blockCount*2u) to store backreferences to iterators
    public:
        _NBL_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(_size_type);

        IteratablePoolAddressAllocator() : base_t() {}
        virtual ~IteratablePoolAddressAllocator() {}

        IteratablePoolAddressAllocator(void* reservedSpc, _size_type addressOffsetToApply, _size_type alignOffsetNeeded, _size_type maxAllocatableAlignment, _size_type bufSz, _size_type blockSz) noexcept :
			base_t(reservedSpc,addressOffsetToApply,alignOffsetNeeded,maxAllocatableAlignment,bufSz,blockSz) {}


        //! When resizing we require that the copying of data buffer has already been handled by the user of the address allocator
        template<typename... Args>
        IteratablePoolAddressAllocator(_size_type newBuffSz, const IteratablePoolAddressAllocator& other, Args&&... args) noexcept :
            base_t(newBuffSz, other, std::forward<Args>(args)...)
        {
            copySupplementaryState(other, newBuffSz);
        }
        
        template<typename... Args>
        IteratablePoolAddressAllocator(_size_type newBuffSz, IteratablePoolAddressAllocator&& other, Args&&... args) noexcept :
            IteratablePoolAddressAllocator(newBuffSz,other,std::forward<Args>(args)...)
        {
            other.base_t::invalidate();
        }

        IteratablePoolAddressAllocator& operator=(IteratablePoolAddressAllocator&& other)
        {
            base_t::operator=(std::move(other));
            return *this;
        }

        //! Functions that actually differ
        inline _size_type        alloc_addr(_size_type bytes, _size_type alignment, _size_type hint=0ull) noexcept
        {
            const _size_type allocatedAddress = base_t::alloc_addr(bytes,alignment,hint);
            if (allocatedAddress!=invalid_address)
            {
                *begin() = allocatedAddress;
                getIteratorOffset(addressToBlockID(allocatedAddress)) = base_t::freeStackCtr;
            }
            return allocatedAddress;
        }

        inline void             free_addr(_size_type addr, _size_type bytes) noexcept
        {
            const _size_type iteratorOffset = getIteratorOffset(addressToBlockID(addr));
            #ifdef _NBL_DEBUG
                assert(iteratorOffset>=base_t::freeStackCtr);
            #endif
            // swap the erased element with either end of the array in the contiguous array
            // not using a swap cause it doesn't matter where the erased element points
            const _size_type otherNodeOffset = *begin();
            reinterpret_cast<_size_type*>(base_t::reservedSpace)[iteratorOffset] = otherNodeOffset;
            // but I need to patch up the back-link of the moved element
            getIteratorOffset(addressToBlockID(otherNodeOffset)) = iteratorOffset;

            base_t::free_addr(addr,bytes);
        }

        // gets a range of all the allocated addresses
        inline const _size_type* begin() const {return &base_t::getFreeStack(base_t::freeStackCtr);}
        inline const _size_type* end() const {return &base_t::getFreeStack(base_t::blockCount);}


        inline _size_type        safe_shrink_size(_size_type sizeBound, _size_type newBuffAlignmentWeCanGuarantee=1u) noexcept
        {
            if (safe_shrink_size_common(sizeBound,newBuffAlignmentWeCanGuarantee))
            {
                assert(begin()!=end()); // we already checked that freeStackCtr>0
                sizeBound = *std::max_element(begin(),end());
            }
            return AddressAllocatorBase<PoolAddressAllocator<_size_type>,_size_type>::safe_shrink_size(sizeBound,newBuffAlignmentWeCanGuarantee);
        }


        static inline _size_type reserved_size(_size_type maxAlignment, _size_type bufSz, _size_type blockSz) noexcept
        {
            _size_type maxBlockCount = bufSz/blockSz;
            return maxBlockCount*sizeof(_size_type)*_size_type(2u);
        }
        static inline _size_type reserved_size(const IteratablePoolAddressAllocator<_size_type>& other, _size_type bufSz) noexcept
        {
            return reserved_size(other.maxRequestableAlignment,bufSz,other.blockSize);
        }

        inline void         reset()
        {
            base_t::reset();
        }
        inline _size_type    max_size() const noexcept
        {
            return base_t::max_size();
        }
        inline _size_type    min_size() const noexcept
        {
            return base_t::min_size();
        }
        inline _size_type    get_free_size() const noexcept
        {
            return base_t::get_free_size();
        }
        inline _size_type    get_allocated_size() const noexcept
        {
            return base_t::get_allocated_size();
        }
        inline _size_type    get_total_size() const noexcept
        {
            return base_t::get_total_size();
        }
        inline _size_type    addressToBlockID(_size_type addr) const noexcept
        {
            return base_t::addressToBlockID(addr);
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
using IteratablePoolAddressAllocatorMT = AddressAllocatorBasicConcurrencyAdaptor<IteratablePoolAddressAllocator<size_type>,RecursiveLockable>;

}
}

#endif

