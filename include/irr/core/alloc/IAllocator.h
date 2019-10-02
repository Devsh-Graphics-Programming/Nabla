// Copyright (C) 2018 Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW Engine"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_I_ALLOCATOR_H_INCLUDED__
#define __IRR_I_ALLOCATOR_H_INCLUDED__

#include "IrrCompileConfig.h"


namespace irr
{
namespace core
{

class IRR_FORCE_EBO IAllocator
{
    public:
        virtual         ~IAllocator() {}

        virtual void*   allocate(size_t n, void* hint=nullptr) noexcept = 0;
        virtual void    deallocate(void* p, size_t n) noexcept = 0;

        virtual size_t  maxsize() const noexcept = 0;
};


template <class Alloc>
class IRR_FORCE_EBO IAllocatorAdaptor final : private Alloc, public IAllocator
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

#endif // __IRR_I_ALLOCATOR_H_INCLUDED__
