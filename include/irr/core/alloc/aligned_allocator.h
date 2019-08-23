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
        typedef size_t	size_type;
        typedef T*		pointer;

        template< class U> struct rebind { typedef aligned_allocator<U,overAlign> other; };


        aligned_allocator() {}
        virtual ~aligned_allocator() {}
        template<typename U, size_t _align = overAlign>
        aligned_allocator(const aligned_allocator<U,_align>& other) {}
        template<typename U, size_t _align = overAlign>
        aligned_allocator(aligned_allocator<U,_align>&& other) {}


        inline typename aligned_allocator::pointer  allocate(   size_type n, size_type alignment,
                                                                const void* hint=nullptr) noexcept
        {
            if (n==0)
                return nullptr;

            void* retval = _IRR_ALIGNED_MALLOC(n*sizeof(T),alignment);
            //printf("Alloc'ed %p with %d\n",retval,n*sizeof(T));
            return reinterpret_cast<typename aligned_allocator::pointer>(retval);
        }
        inline typename aligned_allocator::pointer  allocate(   size_type n, const void* hint=nullptr) noexcept
        {
            return allocate(n,overAlign,hint);
        }

		inline void                                 deallocate(	typename aligned_allocator::pointer p) noexcept
		{
			//printf("Freed %p\n",p);
			//_IRR_ALIGNED_FREE(const_cast<std::remove_const<T>::type*>(p));
			_IRR_ALIGNED_FREE(p);
		}
		inline void                                 deallocate(	typename aligned_allocator::pointer p, size_type n) noexcept
		{
			deallocate(p);
		}

        template<typename U, size_t _align>
        inline bool                                 operator!=(const aligned_allocator<U,_align>& other) const noexcept
        {
            return false;
        }
        template<typename U, size_t _align>
        inline bool                                 operator==(const aligned_allocator<U,_align>& other) const noexcept
        {
            return true;
        }
};

} // end namespace core
} // end namespace irr

#endif

