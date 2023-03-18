// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_ADDRESS_ALLOCATOR_BASE_H_INCLUDED__
#define __NBL_CORE_ADDRESS_ALLOCATOR_BASE_H_INCLUDED__

#include "nbl/core/memory/memory.h"
#include "nbl/core/alloc/address_allocator_traits.h"

namespace nbl
{
namespace core
{

    #define _NBL_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(SIZE_TYPE) \
            typedef SIZE_TYPE                                   size_type;\
            typedef typename std::make_signed<size_type>::type  difference_type;\
            typedef uint8_t*                                    ubyte_pointer;\
            static constexpr size_type                          invalid_address = nbl::core::address_type_traits<size_type>::invalid_address

    template<typename CRTP, typename _size_type>
    class AddressAllocatorBase
    {
        public:
            _NBL_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(_size_type);

            AddressAllocatorBase() :
                reservedSpace(nullptr), addressOffset(invalid_address), alignOffset(invalid_address),
                maxRequestableAlignment(invalid_address), combinedOffset(invalid_address) {}
            


            inline _size_type           max_alignment() const noexcept
            {
                return maxRequestableAlignment;
            }

            inline _size_type           get_align_offset() const noexcept
            {
                return alignOffset;
            }

            inline _size_type           get_combined_offset() const noexcept
            {
                return combinedOffset;
            }

            // usage: for resizeable allocators built on top of non-resizeable ones (that resize via the CRTP constructor)
            // we usually call this before attempting to create the memory for the resized allocator,
            // so we don't know exactly what the addressOffset or alignOffset will be
            inline _size_type           safe_shrink_size(_size_type sizeBound, _size_type newBuffAlignmentWeCanGuarantee=1u) const noexcept
            {
                if (newBuffAlignmentWeCanGuarantee<maxRequestableAlignment)
                    return sizeBound+maxRequestableAlignment-newBuffAlignmentWeCanGuarantee;
                else if (sizeBound)
                    return sizeBound;
                else
                    return std::max(maxRequestableAlignment,newBuffAlignmentWeCanGuarantee);
            }
        protected:
            virtual ~AddressAllocatorBase() {}

            AddressAllocatorBase(void* reservedSpc, _size_type addressOffsetToApply, _size_type alignOffsetNeeded, _size_type maxAllocatableAlignment) :
                    reservedSpace(reservedSpc), addressOffset(addressOffsetToApply), alignOffset(alignOffsetNeeded),
                    maxRequestableAlignment(maxAllocatableAlignment), combinedOffset(addressOffset+alignOffset)
            {
                #ifdef _NBL_DEBUG
                     // pointer to reserved memory has to be aligned to SIMD types!
                    assert((reinterpret_cast<size_t>(reservedSpace)&(_NBL_SIMD_ALIGNMENT-1u))==0ull);
                    assert(maxAllocatableAlignment);
                    assert(core::isPoT(maxRequestableAlignment)); // this is not a proper alignment value
                #endif // _NBL_DEBUG
            }
            AddressAllocatorBase(CRTP&& other, void* newReservedSpc) :
                    reservedSpace(nullptr), addressOffset(invalid_address), alignOffset(invalid_address),
                    maxRequestableAlignment(invalid_address), combinedOffset(invalid_address)
            {
                operator=(std::move(other));
                reservedSpace = newReservedSpc;
                #ifdef _NBL_DEBUG
                    // reserved space has to be aligned at least to SIMD
                    assert((reinterpret_cast<size_t>(reservedSpace)&(_NBL_SIMD_ALIGNMENT-1u))==0ull);
                #endif // _NBL_DEBUG
            }
            AddressAllocatorBase(CRTP&& other, void* newReservedSpc, _size_type newAddressOffset, _size_type newAlignOffset) :
                    AddressAllocatorBase(std::move(other),newReservedSpc)
            {
                addressOffset = newAddressOffset;
                alignOffset = newAlignOffset;
                combinedOffset = addressOffset+alignOffset;
            }
            AddressAllocatorBase(const CRTP& other, void* newReservedSpc) :
                reservedSpace(newReservedSpc), addressOffset(other.addressOffset), alignOffset(other.alignOffset),
                maxRequestableAlignment(other.maxRequestableAlignment), combinedOffset(other.combinedOffset)
            {
                #ifdef _NBL_DEBUG
                    // reserved space has to be aligned at least to SIMD
                    assert((reinterpret_cast<size_t>(reservedSpace)&(_NBL_SIMD_ALIGNMENT-1u))==0ull);
                #endif // _NBL_DEBUG
            }
            AddressAllocatorBase(const CRTP& other, void* newReservedSpc, _size_type newAddressOffset, _size_type newAlignOffset) :
                AddressAllocatorBase(other, newReservedSpc)
            {
                addressOffset = newAddressOffset;
                alignOffset = newAlignOffset;
                combinedOffset = addressOffset+alignOffset;
            }

            inline bool checkResize(_size_type newBuffSz, _size_type newAlignOffset)
            {
                // cannot reallocate the data into smaller storage than is necessary
                if (newBuffSz<safe_shrink_size(0u,maxRequestableAlignment)) return false;
                // need enough space for the new alignment
                if (newAlignOffset>newBuffSz) return false;

                return true;
            }

            inline void*                getAddressOffset() const noexcept {return addressOffset;}

            inline const void*          getReservedSpacePtr() const noexcept {return reservedSpace;}

            AddressAllocatorBase& operator=(AddressAllocatorBase&& other)
            {
                std::swap(reservedSpace,other.reservedSpace);
                std::swap(addressOffset,other.addressOffset);
                std::swap(alignOffset,other.alignOffset);
                std::swap(maxRequestableAlignment,other.maxRequestableAlignment);
                std::swap(combinedOffset,other.combinedOffset);
                return *this;
            }

            // pointer to allocator specific state-keeping data, please note that irrBaW address allocators were designed to allocate memory they can't actually access
            void*           reservedSpace;
            // automatic offset to be added to generated addresses
            _size_type  addressOffset;
            // the positive offset that would have to be applied to `addressOffset`  to make it `maxRequestableAlignment`-aligned to some unknown value
            _size_type  alignOffset;
            // this cannot be deduced from the `addressOffset` , so we let the user specify it
            _size_type  maxRequestableAlignment;

            //internal usage
            _size_type combinedOffset;
    };

}
}

#endif

