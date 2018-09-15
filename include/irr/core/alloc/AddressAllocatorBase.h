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

    template<typename CRTP>
    class AddressAllocatorBase
    {
        public:
            #define DUMMY_DEFAULT_CONSTRUCTOR AddressAllocatorBase() : reservedSpace(nullptr), bufferStart(nullptr) {}
            GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(DUMMY_DEFAULT_CONSTRUCTOR)
            #undef DUMMY_DEFAULT_CONSTRUCTOR

            inline void* getBufferStart() noexcept {return bufferStart;}
        protected:
            virtual ~AddressAllocatorBase() {}

            AddressAllocatorBase(void* reservedSpc, void* buffSz) : reservedSpace(reservedSpc), bufferStart(buffSz)
            {
    #ifdef _DEBUG
                assert((reinterpret_cast<size_t>(reservedSpace)&(_IRR_SIMD_ALIGNMENT-1u))==0ull); // pointer to reserved memory has to be aligned to SIMD types!
    #endif // _DEBUG
            }
            AddressAllocatorBase(CRTP& other, void* newReservedSpc, void* newBuffSz) : reservedSpace(newReservedSpc), bufferStart(newBuffSz)
            {
    #ifdef _DEBUG
                assert((reinterpret_cast<size_t>(reservedSpace)&(_IRR_SIMD_ALIGNMENT-1u))==0ull);
                // cannot reallocate the data into smaller storage than is necessary
                assert(newBuffSz>=other.safe_shrink_size());
                if (other.bufferStart&&bufferStart)
                    memmove(bufferStart,other.bufferStart,newBuffSz);
    #endif // _DEBUG
            }

            void* const reservedSpace;
            void* const bufferStart;
    };

}
}

#endif // __IRR_ADDRESS_ALLOCATOR_BASE_H_INCLUDED__

