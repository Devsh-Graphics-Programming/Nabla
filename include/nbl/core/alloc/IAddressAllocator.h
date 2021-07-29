// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_I_ADDRESS_ALLOCATOR_H_INCLUDED__
#define __NBL_CORE_I_ADDRESS_ALLOCATOR_H_INCLUDED__


#include "nbl/core/decl/BaseClasses.h"
#include "nbl/core/alloc/address_allocator_traits.h"

namespace nbl
{
namespace core
{

class NBL_FORCE_EBO IAddressAllocator : Interface
{
        _NBL_INTERFACE_CHILD_DEFAULT(IAddressAllocator);
    public:
        virtual size_t              get_real_addr(size_t allocated_addr) const noexcept = 0;

        //!
        /** Warning outAddresses needs to be primed with `invalid_address` values,
        otherwise no allocation happens for elements not equal to `invalid_address`.
        \param hint will be needed for 3D texture tile sub-allocation */
        virtual void                multi_alloc_addr(uint32_t count, size_t* outAddresses, const size_t* bytes,
                                                     const size_t* alignment, const size_t* hint=nullptr) noexcept = 0;
        virtual void                multi_free_addr(uint32_t count, const size_t* addr, const size_t* bytes) noexcept = 0;

        virtual void                reset() noexcept = 0;

        virtual size_t              max_size() const noexcept = 0;
        virtual size_t              min_size() const noexcept = 0;
        virtual size_t              max_alignment() const noexcept = 0;

        virtual size_t              safe_shrink_size(size_t sizeBound, size_t newBuffAlignmentWeCanGuarantee=1u) const noexcept =0;
};


template <class AddressAllocator>
class NBL_FORCE_EBO IAddressAllocatorAdaptor final : private AddressAllocator, public IAddressAllocator
{
        inline AddressAllocator&    getBaseRef() noexcept {return static_cast<AddressAllocator&>(*this);}
    public:
        typedef address_allocator_traits<AddressAllocator> traits;

        using AddressAllocator::AddressAllocator;
        virtual                     ~IAddressAllocatorAdaptor() {}

        inline virtual size_t       get_real_addr(size_t allocated_addr) const noexcept override {return traits::get_real_addr(getBaseRef(),allocated_addr);}

        //!
        /** Warning outAddresses needs to be primed with `invalid_address` values,
        otherwise no allocation happens for elements not equal to `invalid_address`. */
        inline virtual void         multi_alloc_addr(uint32_t count, size_t* outAddresses, const size_t* bytes,
                                                     const size_t* alignment, const size_t* hint=nullptr) noexcept override
        {
            traits::multi_alloc_addr(getBaseRef(),count,outAddresses,bytes,alignment,hint);
        }
        inline virtual void         multi_free_addr(uint32_t count, const size_t* addr, const size_t* bytes) noexcept override
        {
            traits::multi_free_addr(getBaseRef(),count,addr,bytes);
        }

        inline virtual void         reset() noexcept override {getBaseRef().reset();}

        inline virtual size_t       max_size() const noexcept override {return getBaseRef().max_size();}

        inline virtual size_t       min_size() const noexcept override {return getBaseRef().min_size();}

        inline virtual size_t       max_alignment() const noexcept override {return getBaseRef().max_alignment();}

        inline virtual size_t       safe_shrink_size(size_t byteBound, size_t newBuffAlignmentWeCanGuarantee=1u) const noexcept override
        {
            return getBaseRef().safe_shrink_size(byteBound,newBuffAlignmentWeCanGuarantee);
        }

        template<typename... Args>
        static inline size_t        reserved_size(const Args&... args) noexcept
        {
            return AddressAllocator::reserved_size(args...);
        }
};

}
}

#endif

