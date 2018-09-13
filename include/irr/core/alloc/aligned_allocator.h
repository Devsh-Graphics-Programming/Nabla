// Copyright (C) 2018 Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW Engine"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_ALIGNED_ALLOCATOR_H_INCLUDED__
#define __IRR_ALIGNED_ALLOCATOR_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "irr/core/memory/memory.h"
#include "irr/core/alloc/AllocatorTrivialBases.h"

namespace irr
{
namespace core
{

template <class T, size_t overAlign=_IRR_DEFAULT_ALIGNMENT(T)>
class IRR_FORCE_EBO aligned_allocator : public irr::core::AllocatorTrivialBase<T>
{
    public:
        typedef size_t      size_type;
        typedef ptrdiff_t   difference_type;

        typedef uint8_t*    ubyte_pointer;

        template< class U> struct rebind { typedef aligned_allocator<U,overAlign> other; };


        GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(aligned_allocator() {})
        virtual ~aligned_allocator() {}
        template<typename U, size_t _align = overAlign>
        aligned_allocator(const aligned_allocator<U,_align>& other) {}
        template<typename U, size_t _align = overAlign>
        aligned_allocator(aligned_allocator<U,_align>&& other) {}


        static inline typename aligned_allocator::pointer   allocate(   size_type n, size_type alignment,
                                                                        typename aligned_allocator::const_void_pointer hint=nullptr) noexcept
        {
            if (n==0)
                return nullptr;

            return reinterpret_cast<typename aligned_allocator::pointer>(_IRR_ALIGNED_MALLOC(n*sizeof(T),alignment));
        }
        inline typename aligned_allocator::pointer          allocate(   size_type n, typename aligned_allocator::const_void_pointer hint=nullptr) noexcept
        {
            return allocate(n,overAlign,hint);
        }

        static inline void                                  deallocate( typename aligned_allocator::pointer p) noexcept
        {
            _IRR_ALIGNED_FREE(p);
        }
        inline void                                         deallocate( typename aligned_allocator::pointer p, size_type n) noexcept
        {
            deallocate(p);
        }

        template<typename U, size_t _align>
        inline bool                                 operator!=(const aligned_allocator<U,_align>& other) noexcept
        {
            return false;
        }
        template<typename U, size_t _align>
        inline bool                                 operator==(const aligned_allocator<U,_align>& other) noexcept
        {
            return true;
        }
};

} // end namespace core
} // end namespace irr

#endif

