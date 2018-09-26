// Copyright (C) 2018 Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW Engine"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_I_ADDRESS_ALLOCATOR_H_INCLUDED__
#define __IRR_I_ADDRESS_ALLOCATOR_H_INCLUDED__


#include "irr/core/alloc/address_allocator_traits.h"

namespace irr
{
namespace core
{

class IRR_FORCE_EBO IAddressAllocator
{
    public:
        virtual                     ~IAddressAllocator() {}

        virtual size_t              get_real_addr(size_t allocated_addr) const noexcept = 0;

        //! \param hint will be needed for 3D texture tile sub-allocation
        virtual void                multi_alloc_addr(size_t* outAddresses, uint32_t count, const size_t* bytes,
                                                     const size_t* alignment, const size_t* hint=nullptr) noexcept = 0;
        virtual void                multi_free_addr(uint32_t count, const size_t* addr, const size_t* bytes) noexcept = 0;
        virtual size_t              alloc_addr(size_t bytes, size_t alignment, size_t hint=0ull) noexcept = 0;
        virtual void                free_addr(size_t addr, size_t bytes) noexcept = 0;

        virtual void                reset() noexcept = 0;

        virtual size_t              max_size() const noexcept = 0;
        virtual size_t              max_alignment() const noexcept = 0;

        virtual size_t              safe_shrink_size(size_t byteBound=0u) const noexcept =0;
};


template <class AddressAllocator>
class IRR_FORCE_EBO IAddressAllocatorAdaptor final : public address_allocator_traits<AddressAllocator>, public IAddressAllocator
{
        inline AddressAllocator&    getBaseRef() noexcept {return static_cast<AddressAllocator&>(*this);}
    public:
        typedef address_allocator_traits<AddressAllocator> traits;

        using AddressAllocator::AddressAllocator;
        virtual                     ~IAddressAllocatorAdaptor() {}

        inline virtual size_t       get_real_addr(size_t allocated_addr) const noexcept {return traits::get_real_addr(getBaseRef(),allocated_addr);}


        inline virtual void         multi_alloc_addr(size_t* outAddresses, uint32_t count, const size_t* bytes,
                                                     const size_t* alignment, const size_t* hint=nullptr) noexcept
        {
            traits::multi_alloc_addr(getBaseRef(),outAddresses,count,bytes,alignment,hint);
        }
        inline virtual void         multi_free_addr(uint32_t count, const size_t* addr, const size_t* bytes) noexcept
        {
            traits::multi_free_addr(getBaseRef(),count,addr,bytes);
        }

        inline virtual size_t       alloc_addr(size_t bytes, size_t alignment, size_t hint=0ull) noexcept
        {
            return getBaseRef().alloc_addr(bytes,alignment,hint);
        }
        inline virtual void         free_addr(size_t addr, size_t bytes) noexcept
        {
            getBaseRef().free_addr(addr,bytes);
        }

        inline virtual void         reset() noexcept {getBaseRef().reset();}

        inline virtual size_t       max_size() const noexcept {return getBaseRef().max_size();}

        inline virtual size_t       max_alignment() const noexcept {return getBaseRef().max_alignment();}

        inline virtual size_t       safe_shrink_size(size_t byteBound=0u) const noexcept {return getBaseRef().safe_shrink_size(byteBound);}

        template<typename... Args>
        static inline size_t        reserved_size(size_t bufSz, const Args&... args) noexcept
        {
            return AddressAllocator::reserved_size(bufSz,args...);
        }
        static inline size_t        reserved_size(size_t bufSz, const AddressAllocator& other) noexcept
        {
            return AddressAllocator::reserved_size(bufSz,other);
        }
};

}
}

#endif // __IRR_I_ADDRESS_ALLOCATOR_H_INCLUDED__

