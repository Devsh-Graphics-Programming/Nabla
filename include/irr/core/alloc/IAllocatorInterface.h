// Copyright (C) 2018 Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW Engine"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_I_ALLOCATOR_INTERFACE_H_INCLUDED__
#define __IRR_I_ALLOCATOR_INTERFACE_H_INCLUDED__

#include "IrrCompileConfig.h"


namespace irr
{
namespace core
{

class IAllocatorInterface
{
    public:
        virtual         ~IAllocatorInterface() {}

        virtual void*   allocate(size_t n, void* hint=nullptr) noexcept = 0;
        virtual void    deallocate(void* p, size_t n) noexcept = 0;

        virtual size_t  max_size() noexcept = 0;
};


template <class Alloc>
class AllocatorInterfaceAdaptor final : private Alloc, public IAllocatorInterface
{
        inline Alloc& getBaseRef() noexcept {return static_cast<Alloc&>(*this);}
    public:
        using Alloc::Alloc;

        virtual ~AllocatorInterfaceAdaptor() {}

        virtual void*   allocate(size_t n, void* hint=nullptr) noexcept {return getBaseRef().allocate(n,hint);}
        virtual void    deallocate(void* p, size_t n) noexcept
        {
            getBaseRef().deallocate(static_cast<typename Alloc::pointer>(p),n);
        }

        virtual size_t  max_size() noexcept {return getBaseRef().max_size();}
};

}
}

#endif // __IRR_I_ALLOCATOR_INTERFACE_H_INCLUDED__
