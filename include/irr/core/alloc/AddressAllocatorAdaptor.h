// Copyright (C) 2018 Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW Engine"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_ADDRESS_ALLOCATOR_ADAPTOR_H_INCLUDED__
#define __IRR_ADDRESS_ALLOCATOR_ADAPTOR_H_INCLUDED__

#include "IrrCompileConfig.h"

#include "irr/core/alloc/AllocatorTrivialBases.h"

namespace irr
{
namespace core
{

//! T is element type, S is the AddressAllocator class
template<typename T, class S>
class AddressAllocatorAdaptor : public AllocatorTrivialBase<T>
{
        template<typename U, class _S>
        friend class    AddressAllocatorAdaptor; // slightly overly friendly

        S* const                            state;
        typedef address_allocator_traits<S> traits;
    public:
        template< class U > struct rebind { typedef AddressAllocatorAdaptor<U,S> other; };


        GCC_CONSTRUCTOR_INHERITANCE_BUG_WORKAROUND(AddressAllocatorAdaptor() : state(nullptr) {})
        virtual ~AddressAllocatorAdaptor()
        {
            if (state)
                state->drop();
        }

        AddressAllocatorAdaptor(S* allocatorState) : state(allocatorState)
        {
            state->grab();
        }
        AddressAllocatorAdaptor(const AddressAllocatorAdaptor& other) : state(other.state)
        {
            state->grab();
        }
        AddressAllocatorAdaptor(AddressAllocatorAdaptor&& other) : state(other.state)
        {
            other.state = nullptr;
        }
        template<typename U>
        AddressAllocatorAdaptor(const AddressAllocatorAdaptor<U,S>& other) : state(other.state)
        {
            state->grab();
        }
        template<typename U>
        AddressAllocatorAdaptor(AddressAllocatorAdaptor<U,S>&& other) : state(other.state)
        {
            other.state = nullptr;
        }


        inline typename AddressAllocatorAdaptor::pointer    allocate(   typename S::size_type n,
                                                                        typename AddressAllocatorAdaptor::const_void_pointer hint=nullptr) noexcept
        {
            n *= sizeof(T);
            typename S::size_type align = alignof(T);
            typename S::size_type addr = S::invalid_address;
            traits::multi_alloc_addr(*state,1u,&addr,&n,&align);
            if (addr==S::invalid_address)
                return nullptr;

            return reinterpret_cast<typename AddressAllocatorAdaptor::pointer>(state->getBufferStart()+addr);
        }

        inline void                                         deallocate( typename AddressAllocatorAdaptor::pointer p,
                                                                        typename S::size_type n) noexcept
        {
            n *= sizeof(T);
            typename S::size_type addr = reinterpret_cast<typename S::ubyte_pointer>(p)-state->getBufferStart();
            traits::multi_free_addr(*state,1u,&addr,&n);
        }

        //! Conservative estimate, max_size() gives largest size we are sure to be able to allocate
        inline typename S::size_type                        max_size() const noexcept
        {
            return state->max_size()/sizeof(T);
        }


        template<typename U>
        inline bool                                         operator!=( const AddressAllocatorAdaptor<U,S>& other) const noexcept
        {
            return state!=other.state;
        }
        template<typename U>
        inline bool                                         operator==( const AddressAllocatorAdaptor<U,S>& other) const noexcept
        {
            return !(operator!=(other));
        }
};

}
}

#endif // __IRR_ADDRESS_ALLOCATOR_ADAPTOR_H_INCLUDED__
