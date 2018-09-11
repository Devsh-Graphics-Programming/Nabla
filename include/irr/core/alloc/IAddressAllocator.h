// Copyright (C) 2018 Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW Engine"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_I_ADDRESS_ALLOCATOR_H_INCLUDED__
#define __IRR_I_ADDRESS_ALLOCATOR_H_INCLUDED__

#include "IrrCompileConfig.h"


namespace irr
{
namespace core
{

class IAddressAllocator
{
    public:
        virtual                     ~IAddressAllocator() {}

        //! \param hint will be needed for 3D texture tile sub-allocation
        virtual size_t              alloc_addr(size_t bytes, size_t alignment, size_t hint=0ull) noexcept = 0;
        virtual void                free_addr(size_t addr, size_t bytes) noexcept = 0;

        virtual void                reset() noexcept = 0;

        virtual size_t              max_size() noexcept = 0;
};


template <class AddressAllocator>
class IAddressAllocatorAdaptor final : private AddressAllocator, public IAddressAllocator
{
        inline AddressAllocator&    getBaseRef() noexcept {return static_cast<AddressAllocator&>(*this);}
    public:
        using AddressAllocator::AddressAllocator;

        virtual                     ~IAddressAllocatorAdaptor() {}

        inline virtual size_t       alloc_addr(size_t bytes, size_t alignment, size_t hint=0ull) noexcept
        {
            return getBaseRef().alloc_addr(bytes,alignment,hint);
        }
        inline virtual void         free_addr(size_t addr, size_t bytes) noexcept
        {
            getBaseRef().free_addr(addr,bytes);
        }

        inline virtual void         reset() noexcept {getBaseRef().reset();}

        inline virtual size_t       max_size() noexcept {return getBaseRef().max_size();}
};

}
}

#endif // __IRR_I_ADDRESS_ALLOCATOR_H_INCLUDED__

