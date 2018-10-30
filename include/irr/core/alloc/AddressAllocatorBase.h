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
            static constexpr size_type                          invalid_address = address_type_traits<size_type>::invalid_address

    template<typename CRTP, typename _size_type>
    class AddressAllocatorBase
    {
        public:
            #define DUMMY_DEFAULT_CONSTRUCTOR AddressAllocatorBase() : reservedSpace(nullptr), bufferStart(nullptr), maxRequestableAlignment(0u), alignOffset(0u) {}
            GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(DUMMY_DEFAULT_CONSTRUCTOR)
            #undef DUMMY_DEFAULT_CONSTRUCTOR

            inline void*                getBufferStart() const noexcept {return bufferStart;}

            inline const void*          getReservedSpacePtr() const noexcept {return reservedSpace;}


            inline _size_type           max_alignment() const noexcept
            {
                return maxRequestableAlignment;
            }

            inline _size_type           get_align_offset() const noexcept
            {
                return alignOffset;
            }

            inline _size_type           safe_shrink_size(_size_type bound, _size_type newBuffAlignmentWeCanGuarantee=1u) const noexcept
            {
                if (newBuffAlignmentWeCanGuarantee<maxRequestableAlignment)
                    return bound+maxRequestableAlignment-newBuffAlignmentWeCanGuarantee;
                else if (bound)
                    return bound;
                else
                    return std::max(maxRequestableAlignment,newBuffAlignmentWeCanGuarantee);
            }


            static inline _size_type    aligned_start_offset(_size_type globalBufferStartAddress, _size_type maxAlignment)
            {
                _size_type remainder = globalBufferStartAddress&(maxAlignment-1u);
                return (globalBufferStartAddress-remainder)&(maxAlignment-1u);
            }
        protected:
            virtual ~AddressAllocatorBase() {}

            AddressAllocatorBase(void* reservedSpc, void* buffStart, _size_type maxAllocatableAlignment) :
                    reservedSpace(reservedSpc), bufferStart(buffStart), maxRequestableAlignment(maxAllocatableAlignment),
                    alignOffset(aligned_start_offset(reinterpret_cast<size_t>(bufferStart),maxRequestableAlignment))
            {
    #ifdef _DEBUG
                assert((reinterpret_cast<size_t>(reservedSpace)&(_IRR_SIMD_ALIGNMENT-1u))==0ull); // pointer to reserved memory has to be aligned to SIMD types!
                assert(core::isPoT(maxRequestableAlignment)); // this is not a proper alignment value
    #endif // _DEBUG
            }
            AddressAllocatorBase(const CRTP& other, void* newReservedSpc, void* newBuffStart, size_t newBuffSz) :
                    reservedSpace(newReservedSpc), bufferStart(newBuffStart), maxRequestableAlignment(other.maxRequestableAlignment),
                    alignOffset(aligned_start_offset(reinterpret_cast<size_t>(bufferStart),maxRequestableAlignment))
            {
    #ifdef _DEBUG
                assert((reinterpret_cast<size_t>(reservedSpace)&(_IRR_SIMD_ALIGNMENT-1u))==0ull);
                // cannot reallocate the data into smaller storage than is necessary
                assert(newBuffSz>=other.safe_shrink_size(0u,maxRequestableAlignment));
                assert(alignOffset<newBuffSz);
    #endif // _DEBUG
            }

            AddressAllocatorBase& operator=(AddressAllocatorBase&& other)
            {
                reservedSpace           = other.reservedSpace;
                bufferStart             = other.bufferStart;
                maxRequestableAlignment = other.maxRequestableAlignment;
                alignOffset             = other.alignOffset;
                return *this;
            }


            void*       reservedSpace;
            void*       bufferStart;
            _size_type  maxRequestableAlignment;
            _size_type  alignOffset;
    };

}
}

#endif // __IRR_ADDRESS_ALLOCATOR_BASE_H_INCLUDED__

