// Copyright (C) 2018 Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW Engine"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_ADDRESS_ALLOCATOR_BASE_H_INCLUDED__
#define __IRR_ADDRESS_ALLOCATOR_BASE_H_INCLUDED__

#include "irr/core/memory/memory.h"
#include "irr/core/alloc/address_allocator_traits.h"

namespace irr
{
namespace core
{

    #define _IRR_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(SIZE_TYPE) \
            typedef SIZE_TYPE                                   size_type;\
            typedef typename std::make_signed<size_type>::type  difference_type;\
            typedef uint8_t*                                    ubyte_pointer;\
            static constexpr size_type                          invalid_address = irr::core::address_type_traits<size_type>::invalid_address

    template<typename CRTP, typename _size_type>
    class AddressAllocatorBase
    {
        public:
            _IRR_DECLARE_ADDRESS_ALLOCATOR_TYPEDEFS(_size_type);

            #define DUMMY_DEFAULT_CONSTRUCTOR AddressAllocatorBase() :\
                reservedSpace(nullptr), addressOffset(invalid_address), alignOffset(invalid_address),\
                maxRequestableAlignment(invalid_address), combinedOffset(invalid_address) {}
            GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(DUMMY_DEFAULT_CONSTRUCTOR)
            #undef DUMMY_DEFAULT_CONSTRUCTOR


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
                #ifdef _IRR_DEBUG
                     // pointer to reserved memory has to be aligned to SIMD types!
                    assert((reinterpret_cast<size_t>(reservedSpace)&(_IRR_SIMD_ALIGNMENT-1u))==0ull);
                    assert(maxAllocatableAlignment);
                    assert(core::isPoT(maxRequestableAlignment)); // this is not a proper alignment value
                #endif // _IRR_DEBUG
            }
            AddressAllocatorBase(CRTP&& other, void* newReservedSpc) :
                    reservedSpace(nullptr), addressOffset(invalid_address), alignOffset(invalid_address),
                    maxRequestableAlignment(invalid_address), combinedOffset(invalid_address)
            {
                operator=(std::move(other));
                reservedSpace = newReservedSpc;
                #ifdef _IRR_DEBUG
                    // reserved space has to be aligned at least to SIMD
                    assert((reinterpret_cast<size_t>(reservedSpace)&(_IRR_SIMD_ALIGNMENT-1u))==0ull);
                #endif // _IRR_DEBUG
            }
            AddressAllocatorBase(CRTP&& other, void* newReservedSpc, _size_type newAddressOffset, _size_type newAlignOffset) :
                    AddressAllocatorBase(std::move(other),newReservedSpc)
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

#endif // __IRR_ADDRESS_ALLOCATOR_BASE_H_INCLUDED__

