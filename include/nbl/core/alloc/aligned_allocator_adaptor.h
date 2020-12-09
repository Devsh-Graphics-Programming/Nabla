// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_ALIGNED_ALLOCATOR_ADAPTOR_H_INCLUDED__
#define __NBL_CORE_ALIGNED_ALLOCATOR_ADAPTOR_H_INCLUDED__

#include <memory>

#include "nbl/core/memory/memory.h"

namespace nbl
{
namespace core
{

template <class Alloc, size_t overAlign=_NBL_DEFAULT_ALIGNMENT(typename std::allocator_traits<Alloc>::value_type)>
class NBL_FORCE_EBO aligned_allocator_adaptor : public Alloc
{
    public:
        typedef std::allocator_traits<Alloc>                    traits;

        template< class U > struct rebind
        {
            typedef aligned_allocator_adaptor<typename Alloc::template rebind<U>::other,overAlign> other;
        };

        constexpr static size_t m_type_align                    = alignof(typename traits::value_type);
        constexpr static size_t m_align                         = overAlign>m_type_align ? overAlign:m_type_align;
        constexpr static size_t m_align_minus_1                 = m_align-1u;

        typedef typename traits::template rebind_alloc<uint8_t> ubyte_alloc;
        typedef typename ubyte_alloc::pointer                   ubyte_ptr;

    private:
        // better and faster than boost for stateful allocators (no construct/destruct of rebound allocator on every alloc/dealloc)
        ubyte_alloc                                             cachedAllocator{static_cast<const Alloc&>(*this)};
    public:
        virtual ~aligned_allocator_adaptor() {}
        //constructors
        using Alloc::Alloc;

        template<class Alloc2, size_t Alignment2>
        inline aligned_allocator_adaptor(const aligned_allocator_adaptor<Alloc2,Alignment2>& other) noexcept :
                                                                            Alloc(static_cast<const Alloc&>(other)) {}

        template<class Alloc2, size_t Alignment2>
        inline aligned_allocator_adaptor(aligned_allocator_adaptor<Alloc2,Alignment2>&& other) noexcept :
                                                            Alloc(std::forward(static_cast<Alloc&&>(other))) {}

        //
        inline typename Alloc::pointer  allocate(typename Alloc::size_type n, typename Alloc::const_void_pointer hint=nullptr)
        {
            if (n==0)
                return nullptr;

            const typename Alloc::size_type requestBytes = n*sizeof(typename Alloc::value_type);
            auto requestBytesPadded = requestBytes+m_align_minus_1;

            typename ubyte_alloc::const_void_pointer h = nullptr;
            if (hint)
                h = *(static_cast<const ubyte_ptr*>(hint) - 1u);

            ubyte_ptr const actualAlloc = cachedAllocator.allocate(sizeof(ubyte_ptr) + requestBytesPadded,h);
            void* retval = actualAlloc + sizeof(ubyte_ptr);
            (void)std::align(m_align, requestBytes, retval, requestBytesPadded);
            ::new(static_cast<void*>(static_cast<ubyte_ptr*>(retval) - 1u)) ubyte_ptr(actualAlloc);
            return static_cast<typename Alloc::pointer>(retval);
        }

        inline void                     deallocate(typename Alloc::pointer p, typename Alloc::size_type n)
        {
            if (!p)
                return;

            const typename Alloc::size_type requestBytes = n*sizeof(typename Alloc::value_type);
            ubyte_ptr* ptr = reinterpret_cast<ubyte_ptr*>(p) - 1u;
            ubyte_ptr actualAlloc = *ptr;
            ptr->~ubyte_ptr();
            cachedAllocator.deallocate(actualAlloc, sizeof(ubyte_ptr) + requestBytes + m_align_minus_1);
        }

        template<size_t _align>
        inline bool                                 operator!=(const aligned_allocator_adaptor<Alloc,_align>& other) const noexcept
        {
            return false;
        }
        template<size_t _align>
        inline bool                                 operator==(const aligned_allocator_adaptor<Alloc,_align>& other) const noexcept
        {
            return true;
        }
};

} // end namespace core
} // end namespace nbl

#endif


