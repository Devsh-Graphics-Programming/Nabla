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

        S* const        state;
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
            typename S::size_type addr = state->alloc_addr(n*sizeof(T),alignof(T));
            if (addr==S::invalid_address)
                return nullptr;

            return reinterpret_cast<typename AddressAllocatorAdaptor::pointer>(state->getBufferStart()+addr);
        }

        inline void                                         deallocate( typename AddressAllocatorAdaptor::pointer p,
                                                                        typename S::size_type n) noexcept
        {
            state->free_addr(reinterpret_cast<typename S::ubyte_pointer>(p)-state->getBufferStart(),n*sizeof(T));
        }

        inline typename S::size_type                        max_size() noexcept
        {
            return state->max_size()/sizeof(T);
        }


        template<typename U>
        inline bool                                         operator!=( const AddressAllocatorAdaptor<U,S>& other) noexcept
        {
            return state!=other.state;
        }
        template<typename U>
        inline bool                                         operator==( const AddressAllocatorAdaptor<U,S>& other) noexcept
        {
            return !(operator!=(other));
        }
};

}
}

#endif // __IRR_ADDRESS_ALLOCATOR_ADAPTOR_H_INCLUDED__
