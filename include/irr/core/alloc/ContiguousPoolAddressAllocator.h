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
        // based on the assumption of 32kb L1, and trying not to use all of it in `multi_free`
        static constexpr uint32_t maxMultiOps = 16u*1024u/sizeof(_size_type);

        #define DUMMY_DEFAULT_CONSTRUCTOR ContiguousPoolAddressAllocator() : redirToAddress(nullptr), redirToMemory(nullptr), addressesAllocated(invalid_address) {}
        GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(DUMMY_DEFAULT_CONSTRUCTOR)
        #undef DUMMY_DEFAULT_CONSTRUCTOR

        virtual ~ContiguousPoolAddressAllocator() {}

        template<typename... ExtraArgs>
        ContiguousPoolAddressAllocator( void* reservedSpc, _size_type addressOffsetToApply, _size_type alignOffsetNeeded, _size_type maxAllocatableAlignment,
                                                                size_type bufSz, size_type blockSz, void* dataBuf) noexcept :
                    Base(reservedSpc,addressOffsetToApply,alignOffsetNeeded,maxAllocatableAlignment,bufSz,blockSz),
                    redirToAddress(Base::freeStack+Base::reserved_size(*this,bufSz)/sizeof(size_type)),
                    redirToMemory(redirToAddress+Base::blockCount), addressesAllocated(0u), dataBuffer(dataBuf)
        {
            selfOnlyReset();
        }

        //! When resizing we require that the copying of data buffer has already been handled by the user of the address allocator even if `supportsNullBuffer==true`
        template<typename... Args>
        ContiguousPoolAddressAllocator(void* newDataBuffer, _size_type newBuffSz, ContiguousPoolAddressAllocator&& other, Args&&... args) noexcept :
                    Base(setUpRedirectsOutOfOrder(newBuffSz,other,args...),std::move(other),std::forward<Args>(args)...),
                    redirToAddress(nullptr), redirToMemory(nullptr), addressesAllocated(invalid_address), dataBuffer(newDataBuffer)
        {
            std::swap(redirToAddress,other.redirToAddress);
            std::swap(redirToMemory,other.redirToMemory);
            std::swap(addressesAllocated,other.addressesAllocated);
            other.dataBuffer = nullptr;
            #ifdef _IRR_DEBUG
                assert(dataBuffer);
                #ifdef _EXTREME_DEBUG
                validateRedirects();
                #endif // _EXTREME_DEBUG
            #endif // _IRR_DEBUG
        }

        ContiguousPoolAddressAllocator& operator=(ContiguousPoolAddressAllocator&& other)
        {
            static_cast<Base&>(*this) = std::move(other);
            std::swap(redirToAddress,other.redirToAddress);
            std::swap(redirToMemory,other.redirToMemory);
            std::swap(addressesAllocated,other.addressesAllocated);
            std::swap(dataBuffer,other.dataBuffer);
            return *this;
        }

        inline void setDataBufferPtr(void* newDataBuffer) noexcept
        {
            dataBuffer = newDataBuffer;
        }

        // this one differs from Base
        inline size_type        get_real_addr(size_type allocated_addr) const
        {
            return redirToMemory[Base::addressToBlockID(allocated_addr)];
        }

        // extra
        inline void             multi_free_addr(uint32_t count, const size_type* addr, const size_type* bytes) noexcept
        {
            if (count==0)
                return;

            const auto addressLimit = addressesAllocated*Base::blockSize+Base::alignOffset;

            size_type sortedRedirects[maxMultiOps];
            size_type* sortedRedirectsEnd = sortedRedirects;
            for (decltype(count) i=0; i<count; i++)
            {
                auto tmp  = addr[i];
                if (tmp==invalid_address)
                    continue;
                #ifdef _IRR_DEBUG
                    assert(tmp>=Base::combinedOffset);
                #endif // _IRR_DEBUG
                // add allocated address back onto free stack
                Base::freeStack[Base::freeStackCtr++] = tmp;
                auto& redir = redirToMemory[Base::addressToBlockID(tmp)];
                *(sortedRedirectsEnd++) = redir;
                #ifdef _IRR_DEBUG
                    assert(redir!=invalid_address);
                    assert(redir<addressLimit);
                    assert(redirToAddress[Base::addressToBlockID(redir)]==tmp);
                    redir = invalid_address;
                #endif // _IRR_DEBUG
            }
            if (sortedRedirectsEnd==sortedRedirects)
                return;

            // sortedRedirects becomes a list of holes in our contiguous array
            std::make_heap(sortedRedirects,sortedRedirectsEnd);
            std::sort_heap(sortedRedirects,sortedRedirectsEnd);

            #ifndef _IRR_DEBUG
            if (addressesAllocated==count)
            {
                addressesAllocated = 0u;
                return;
            }
            #endif // _IRR_DEBUG

            auto movedRedirStart = redirToAddress+Base::addressToBlockID(sortedRedirects[0]);
            auto movedRangeStart = reinterpret_cast<uint8_t*>(dataBuffer)+(sortedRedirects[0]-Base::combinedOffset);
            for (decltype(count) j=0u; j<count; )
            {
                decltype(count) nextIx = j+1u;
                // after this loop, `nextIx` will be at the first value not adjacent with range starting at j
                size_type nextVal=sortedRedirects[j]+Base::blockSize;
                for (; nextIx<count; nextIx++,nextVal+=Base::blockSize)
                {
                    if (nextVal!=sortedRedirects[nextIx])
                        break;
                }
                // now have to erase the [sortedRegirects[j],nextVal] range
                // shift back references and adjust address mappings
                auto rangeEnd = nextIx<count ? sortedRedirects[nextIx]:addressLimit;
                auto blockRangeEnd = Base::addressToBlockID(rangeEnd);
                for (auto i=Base::addressToBlockID(nextVal); i<blockRangeEnd; i++)
                {
                    auto tmpAddr = redirToAddress[i];
                    *movedRedirStart++ = tmpAddr;
                    redirToMemory[Base::addressToBlockID(tmpAddr)] -= nextIx*Base::blockSize;
                }
                // shift actual memory
                auto rangeLen = rangeEnd-nextVal;
                memmove(movedRangeStart,reinterpret_cast<uint8_t*>(dataBuffer)+(nextVal-Base::combinedOffset),rangeLen);
                movedRangeStart+=rangeLen;
                j = nextIx;
            }
            #ifdef _IRR_DEBUG
                for (auto rangeEnd=redirToAddress+addressesAllocated; movedRedirStart!=rangeEnd; movedRedirStart++)
                    *movedRedirStart = invalid_address;
            #endif // _IRR_DEBUG

            // finslly reduce the count
            addressesAllocated -= count;

            #ifdef _EXTREME_DEBUG
            validateRedirects();
            #endif // _EXTREME_DEBUG
        }

        inline void             free_addr(size_type addr, size_type bytes) noexcept
        {
            multi_free_addr(1u, &addr, &bytes);
        }

        //! non-PoT alignments cannot be guaranteed after a resize or move of the backing buffer
        inline size_type        alloc_addr(size_type bytes, size_type alignment, size_type hint=0ull) noexcept
        {
            auto ID = Base::alloc_addr(bytes,alignment,hint);
            if (ID==invalid_address)
                return invalid_address;

            redirToAddress[addressesAllocated] = ID;
            redirToMemory[Base::addressToBlockID(ID)] = addressesAllocated*Base::blockSize;
            addressesAllocated++;

            return ID;
        }

        inline void             reset()
        {
            Base::reset();
            selfOnlyReset();
        }


        inline size_type        safe_shrink_size(size_type sizeBound, size_type newBuffAlignmentWeCanGuarantee=1u) noexcept
        {
            auto boundByAllocCount = addressesAllocated*Base::blockSize;
            if (sizeBound<boundByAllocCount)
                sizeBound = boundByAllocCount;

            return Base::safe_shrink_size(sizeBound,newBuffAlignmentWeCanGuarantee);
        }

        // dummy needed for some of the higher-up classes, so that arguments to constructor after `maxAlignment` can also be sent to `reserved_size`
        static inline size_type reserved_size(size_type maxAlignment, size_type bufSz, size_type blockSz, void* dummy=nullptr) noexcept
        {
            size_type retval = Base::reserved_size(maxAlignment,bufSz,blockSz);
            size_type maxBlockCount =  bufSz/blockSz;
            return retval+maxBlockCount*sizeof(size_type)*size_type(2u);
        }
        static inline size_type reserved_size(const ContiguousPoolAddressAllocator<_size_type>& other, size_type bufSz) noexcept
        {
            return reserved_size(other.maxRequestableAlignment,bufSz,other.blockSize);
        }


        inline size_type addressToBlockID(size_type addr) const noexcept
        {
            return Base::addressToBlockID(addr);
        }
    protected:
        size_type*  redirToAddress;
        size_type*  redirToMemory;
        size_type   addressesAllocated;
        void*           dataBuffer;
    private:
        inline void selfOnlyReset()
        {
            #ifdef _IRR_DEBUG
                for (size_type i=0ull; i<Base::blockCount; i++)
                {
                    redirToAddress[i] = invalid_address;
                    redirToMemory[i] = invalid_address;
                }
            #endif // _IRR_DEBUG
            addressesAllocated = 0ull;
        }

        #ifdef _EXTREME_DEBUG
        inline void validateRedirects() const
        {
            for (size_type i=0; i<Base::freeStackCtr; i++)
            {
                _IRR_BREAK_IF(Base::freeStack[i]>=Base::blockCount*Base::blockSize+Base::combinedOffset);
                _IRR_BREAK_IF(redirToMemory[Base::addressToBlockID(Base::freeStack[i])]!=invalid_address);
            }

            size_type reportedAllocated=0;
            for (size_type i=0; i<Base::blockCount; i++)
            {
                size_type mem = redirToMemory[i];
                if (mem==invalid_address)
                    continue;
                reportedAllocated++;

                _IRR_BREAK_IF(mem>=addressesAllocated*Base::blockSize+Base::combinedOffset);
                _IRR_BREAK_IF(redirToAddress[Base::addressToBlockID(mem)]!=i*Base::blockSize+Base::combinedOffset);
                for (size_type j=0; j<i; j++)
                    _IRR_BREAK_IF(mem==redirToMemory[j]);
            }
            _IRR_BREAK_IF(addressesAllocated!=reportedAllocated);

            for (size_type i=0; i<addressesAllocated; i++)
            {
                size_type addr = redirToAddress[i];
                _IRR_BREAK_IF(addr==invalid_address);
                _IRR_BREAK_IF(addr>=Base::blockCount*Base::blockSize+Base::combinedOffset);
                _IRR_BREAK_IF(redirToMemory[Base::addressToBlockID(addr)]!=i*Base::blockSize+Base::combinedOffset);
                for (size_type j=0; j<i; j++)
                    _IRR_BREAK_IF(addr==redirToAddress[j]);
            }
            for (size_type i=addressesAllocated; i<Base::blockCount; i++)
                _IRR_BREAK_IF(redirToAddress[i]!=invalid_address);
        }
        #endif // _EXTREME_DEBUG

        inline size_type setUpRedirectsOutOfOrder(size_type newBuffSz, ContiguousPoolAddressAllocator& other, void* newReservedSpc) noexcept
        {
            return setUpRedirectsOutOfOrder(newBuffSz,other,newReservedSpc,other.addressOffset,other.alignOffset);
        }
        inline size_type setUpRedirectsOutOfOrder(size_type newBuffSz, ContiguousPoolAddressAllocator& other, void* newReservedSpc, _size_type newAddressOffset) noexcept
        {
            return setUpRedirectsOutOfOrder(newBuffSz,other,newReservedSpc,newAddressOffset,other.alignOffset);
        }
        inline size_type setUpRedirectsOutOfOrder(size_type newBuffSz, ContiguousPoolAddressAllocator& other, void* newReservedSpc, _size_type newAddressOffset, _size_type newAlignOffset) noexcept
        {
            #ifdef _EXTREME_DEBUG
                other.validateRedirects();
            #endif // _EXTREME_DEBUG

            auto newBlockCount = (newBuffSz-newAlignOffset)/other.blockSize;
            size_type* newRedirToAddress = reinterpret_cast<size_type*>(reinterpret_cast<uint8_t*>(newReservedSpc)+Base::reserved_size(other,newBuffSz));
            size_type* newRedirToMemory = newRedirToAddress+newBlockCount;
            #ifdef _IRR_DEBUG
                for (size_type i=0u; i<newBlockCount; i++)
                {
                    newRedirToAddress[i] = invalid_address;
                    newRedirToMemory[i] = invalid_address;
                    if (i<other.addressesAllocated)
                        continue;
                    if (i<other.blockCount)
                        assert(other.redirToAddress[i]==invalid_address);
                }
            #endif // _IRR_DEBUG

            auto newCombinedOffset = newAddressOffset+newAlignOffset;
            for (size_type i=0u; i<other.addressesAllocated; i++)
            {
                auto addr = other.redirToAddress[i];
                auto addrSansOffset = addr-other.combinedOffset;
                auto addrBlock = addrSansOffset/other.blockSize;
                #ifdef _IRR_DEBUG
                    assert(addr!=invalid_address && addrBlock<other.blockCount);
                #endif // _IRR_DEBUG
                auto memRedir = other.redirToMemory[addrBlock];
                #ifdef _IRR_DEBUG
                    assert(memRedir!=invalid_address);
                #endif // _IRR_DEBUG
                newRedirToAddress[i] = addrSansOffset+newCombinedOffset;
                newRedirToMemory[addrBlock] = memRedir+(newCombinedOffset-other.combinedOffset);
            }
            other.redirToAddress = newRedirToAddress;
            other.redirToMemory = newRedirToMemory;
            return newBuffSz;
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


