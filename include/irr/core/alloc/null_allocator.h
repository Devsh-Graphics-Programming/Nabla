// Copyright (C) 2018 Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW Engine"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_NULL_ALLOCATOR_H_INCLUDED__
#define __IRR_NULL_ALLOCATOR_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "irr/core/memory/memory.h"
#include "irr/core/alloc/AllocatorTrivialBases.h"

namespace irr
{
namespace core
{

template <class T, size_t overAlign=_IRR_DEFAULT_ALIGNMENT(T)>
class IRR_FORCE_EBO null_allocator : public irr::core::AllocatorTrivialBase<T>
{
    public:
        typedef size_t      size_type;
        typedef ptrdiff_t   difference_type;

        typedef uint8_t*    ubyte_pointer;

        template< class U> struct rebind { typedef null_allocator<U,overAlign> other; };


        null_allocator() {}
        virtual ~null_allocator() {}
        template<typename U, size_t _align = overAlign>
        null_allocator(const null_allocator<U,_align>& other) {}
        template<typename U, size_t _align = overAlign>
        null_allocator(null_allocator<U,_align>&& other) {}


        inline typename null_allocator::pointer  allocate(   size_type n, size_type alignment,
                                                                typename null_allocator::const_void_pointer hint=nullptr) noexcept
        {
            return nullptr;
        }
        inline typename null_allocator::pointer  allocate(   size_type n, typename null_allocator::const_void_pointer hint=nullptr) noexcept
        {
            return nullptr;
        }

        inline void                                 deallocate( typename null_allocator::pointer p) noexcept
        {
        }
        inline void                                 deallocate( typename null_allocator::pointer p, size_type n) noexcept
        {
        }

        template<typename U, size_t _align>
        inline bool                                 operator!=(const null_allocator<U,_align>& other) const noexcept
        {
            return false;
        }
        template<typename U, size_t _align>
        inline bool                                 operator==(const null_allocator<U,_align>& other) const noexcept
        {
            return true;
        }
};

} // end namespace core
} // end namespace irr

#endif

