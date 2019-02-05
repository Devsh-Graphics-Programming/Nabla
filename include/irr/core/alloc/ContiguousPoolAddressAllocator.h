// Copyright (C) 2018 Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW Engine"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_CONTIGUOUS_POOL_ADDRESS_ALLOCATOR_H_INCLUDED__
#define __IRR_CONTIGUOUS_POOL_ADDRESS_ALLOCATOR_H_INCLUDED__

#include "IrrCompileConfig.h"

#include <cstring>
#include "irr/core/alloc/PoolAddressAllocator.h"

namespace irr
{
namespace core
{


//! Can only allocate up to a size of a single block, no support for allocations larger than blocksize
template<typename _size_type>
class ContiguousPoolAddressAllocator : protected PoolAddressAllocator<_size_type>
{
    private:
        typedef PoolAddressAllocator<_size_type>    Base;
    public:
        _IRR_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(_size_type);

        static constexpr bool supportsNullBuffer = false;
        static constexpr uint32_t maxMultiOps = 4096u;

        #define DUMMY_DEFAULT_CONSTRUCTOR ContiguousPoolAddressAllocator() : addressRedirects(nullptr), addressesAllocated(0u) {}
        GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(DUMMY_DEFAULT_CONSTRUCTOR)
        #undef DUMMY_DEFAULT_CONSTRUCTOR

        virtual ~ContiguousPoolAddressAllocator() {}

        ContiguousPoolAddressAllocator(void* reservedSpc, void* buffer, size_type maxAllocatableAlignment, size_type bufSz, size_type blockSz) noexcept :
                                PoolAddressAllocator<_size_type>(reservedSpc,buffer,maxAllocatableAlignment,bufSz,blockSz),
                                addressRedirects(reinterpret_cast<size_type*>(Base::reservedSpace)+Base::blockCount)
        {
            selfOnlyReset();
        }

        //! When resizing we require that the copying of data buffer has already been handled by the user of the address allocator even if `supportsNullBuffer==true`
        ContiguousPoolAddressAllocator(const ContiguousPoolAddressAllocator& other, void* newReservedSpc, void* newBuffer, size_type newBuffSz) noexcept :
                                PoolAddressAllocator<_size_type>(other,newReservedSpc,newBuffer,newBuffSz),
                                addressRedirects(reinterpret_cast<size_type*>(Base::reservedSpace)+Base::blockCount),
                                addressesAllocated(other.addressesAllocated)
        {
            for (size_type i=0u; i<Base::blockCount; i++)
            {
                if (i<other.blockCount && other.addressRedirects[i] != invalid_address)
                    addressRedirects[i] = other.addressRedirects[i]+Base::alignOffset-other.alignOffset;
                else
                    addressRedirects[i] = invalid_address;
            }
        }

        ContiguousPoolAddressAllocator& operator=(ContiguousPoolAddressAllocator&& other)
        {
            static_cast<Base&>(*this) = std::move(other);
            addressRedirects = other.addressRedirects;
            addressesAllocated = other.addressesAllocated;


            return *this;
        }

        // extra
        inline size_type        get_real_addr(size_type allocated_addr) const
        {
            return addressRedirects[Base::addressToBlockID(allocated_addr)];
        }

        // extra
        inline void             multi_free_addr(uint32_t count, const size_type* addr, const size_type* bytes) noexcept
        {
            if (count==0)
                return;

            size_type sortedRedirects[maxMultiOps];
            size_type* sortedRedirectsEnd = sortedRedirects;
            for (decltype(count) i=0; i<count; i++)
            {
                auto tmp  = addr[i];
                if (tmp==invalid_address)
                    continue;
#ifdef _DEBUG
                assert(tmp>=Base::alignOffset);
#endif // _DEBUG
                // add allocated address back onto free stack
                reinterpret_cast<size_type*>(Base::reservedSpace)[Base::freeStackCtr++] = tmp;
                auto redir = get_real_addr(tmp);
#ifdef _DEBUG
                assert(redir<addressesAllocated*Base::blockSize);
#endif // _DEBUG
                *(sortedRedirectsEnd++) = redir;
            }
            if (sortedRedirectsEnd==sortedRedirects)
                return;

            // sortedRedirects becomes a list of holes in our contiguous array
            std::make_heap(sortedRedirects,sortedRedirectsEnd);
            std::sort_heap(sortedRedirects,sortedRedirectsEnd);


            // shift redirects
            for (size_t i=0; i<Base::blockCount; i++)
            {
                size_type rfrnc = addressRedirects[i];
                if (rfrnc>=invalid_address)
                    continue;

                // find first contiguous address to be deleted larger or equal to
                size_type* ptr = std::lower_bound(sortedRedirects,sortedRedirectsEnd,rfrnc);
                if (ptr<sortedRedirectsEnd && ptr[0]==rfrnc) // found in hole list
                    addressRedirects[i] = invalid_address;
                else if (ptr>sortedRedirects)
                {
                    size_type difference = ptr-sortedRedirects;
                    addressRedirects[i] -= difference*Base::blockSize;
                }
            }

            if (addressesAllocated==count)
            {
                addressesAllocated = 0u;
                return;
            }

            if (Base::bufferStart)
            {
                decltype(count) nextIx=1;
                decltype(count) j=0;
                while (nextIx<count)
                {
                    size_t len = sortedRedirects[nextIx]-sortedRedirects[j]-Base::blockSize;
                    if (len)
                    {
                        ubyte_pointer oldRedirectedAddress = reinterpret_cast<ubyte_pointer>(Base::bufferStart)+sortedRedirects[j];
                        memmove(oldRedirectedAddress-j*Base::blockSize,oldRedirectedAddress+Base::blockSize,len);
                    }

                    j = nextIx++;
                }
                size_type len = (addressesAllocated-1u)*Base::blockSize-sortedRedirects[j];
                if (len)
                {
                    ubyte_pointer oldRedirectedAddress = reinterpret_cast<ubyte_pointer>(Base::bufferStart)+sortedRedirects[j];
                    memmove(oldRedirectedAddress-j*Base::blockSize,oldRedirectedAddress+Base::blockSize,len);
                }
            }
            addressesAllocated -= count;
        }

        //! non-PoT alignments cannot be guaranteed after a resize or move of the backing buffer
        inline size_type        alloc_addr(size_type bytes, size_type alignment, size_type hint=0ull) noexcept
        {
            if (bytes>Base::blockSize)
                return invalid_address;

            auto ID = Base::alloc_addr(bytes,alignment,hint);
            if (ID==invalid_address)
                return invalid_address;

            addressRedirects[Base::addressToBlockID(ID)] = (addressesAllocated++)*Base::blockSize;

            return ID;
        }

        inline void             reset()
        {
            _IRR_BREAK_IF(true);
            Base::reset();
            selfOnlyReset();
        }


        inline size_type        safe_shrink_size(size_type byteBound=0u, size_type newBuffAlignmentWeCanGuarantee=1u) const noexcept
        {
            auto boundByAllocCount = addressesAllocated*Base::blockSize;
            if (byteBound<boundByAllocCount)
                byteBound = boundByAllocCount;

            size_type newBound = Base::blockCount;
            for (; newBound>byteBound/Base::blockSize; newBound--)
            {
                if (addressRedirects[newBound-1u]<invalid_address)
                    break;
            }
            newBound *= Base::blockSize;
            if (byteBound<newBound)
                byteBound = newBound*Base::blockSize;

            return Base::safe_shrink_size(byteBound,newBuffAlignmentWeCanGuarantee);
        }


        template<typename... Args>
        static inline size_type reserved_size(const Args&... args) noexcept
        {
            return Base::reserved_size(args...)*2ull;
        }


        inline size_type addressToBlockID(size_type addr) const noexcept
        {
            return Base::addressToBlockID(addr);
        }
    protected:
        size_type*                              addressRedirects;
        size_type                               addressesAllocated;
    private:
        inline void selfOnlyReset()
        {
            for (size_type i=0ull; i<Base::blockCount; i++)
                addressRedirects[i] = invalid_address;
            addressesAllocated = 0ull;
        }

        inline void validateRedirects()
        {
            auto freeStack = reinterpret_cast<size_type*>(Base::reservedSpace);
            for (size_type i=0; i<Base::freeStackCtr; i++)
            {
                _IRR_BREAK_IF(freeStack[i]>=Base::blockCount*Base::blockSize+Base::alignOffset);
                _IRR_BREAK_IF(addressRedirects[Base::addressToBlockID(freeStack[i])]!=invalid_address);
            }

            size_type reportedAllocated=0;
            for (size_type i=0; i<Base::blockCount; i++)
            {
                size_type key = addressRedirects[i];
                if (key==invalid_address)
                    continue;
                reportedAllocated++;

                _IRR_BREAK_IF(key>=addressesAllocated*Base::blockSize+Base::alignOffset);
                for (size_type j=0; j<i; j++)
                    _IRR_BREAK_IF(key==addressRedirects[j]);
            }
            _IRR_BREAK_IF(addressesAllocated!=reportedAllocated);
        }
};


// aliases
template<typename size_type>
using ContiguousPoolAddressAllocatorST = ContiguousPoolAddressAllocator<size_type>;

template<typename size_type, class RecursiveLockable>
using ContiguousPoolAddressAllocatorMT = AddressAllocatorBasicConcurrencyAdaptor<ContiguousPoolAddressAllocator<size_type>,RecursiveLockable>;

}
}

#endif // __IRR_CONTIGUOUS_POOL_ADDRESS_ALLOCATOR_H_INCLUDED__


