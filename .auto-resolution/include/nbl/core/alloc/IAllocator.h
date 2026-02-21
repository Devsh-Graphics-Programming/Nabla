// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_I_ALLOCATOR_H_INCLUDED__
#define __NBL_CORE_I_ALLOCATOR_H_INCLUDED__

namespace nbl
{
namespace core
{

class NBL_FORCE_EBO IAllocator
{
    public:
        virtual         ~IAllocator() {}

        virtual void*   allocate(size_t n, void* hint=nullptr) noexcept = 0;
        virtual void    deallocate(void* p, size_t n) noexcept = 0;

        virtual size_t  maxsize() const noexcept = 0;
};


template <class Alloc>
class NBL_FORCE_EBO IAllocatorAdaptor final : private Alloc, public IAllocator
{
        inline Alloc& getBaseRef() noexcept {return static_cast<Alloc&>(*this);}
    public:
        using Alloc::Alloc;

        virtual ~IAllocatorAdaptor() {}

        virtual void*   allocate(size_t n, void* hint=nullptr) noexcept {return getBaseRef().allocate(n,hint);}
        virtual void    deallocate(void* p, size_t n) noexcept
        {
            getBaseRef().deallocate(static_cast<typename Alloc::pointer>(p),n);
        }

        virtual size_t  maxsize() const noexcept {return getBaseRef().maxsize();}
};

}
}

#endif
