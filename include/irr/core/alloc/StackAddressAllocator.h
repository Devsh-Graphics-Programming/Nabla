// Copyright (C) 2018 Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW Engine"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_STACK_ADDRESS_ALLOCATOR_H_INCLUDED__
#define __IRR_STACK_ADDRESS_ALLOCATOR_H_INCLUDED__

#include "IrrCompileConfig.h"

#include "irr/core/alloc/address_allocator_type_traits.h"


namespace irr
{
namespace core
{

//! TODO: Stack Allocator with alignment is not that easy, needs extra state (and minimum allocation size)

/**
template<typename _size_type>
class StackAddressAllocator;


// aliases no point for a Multithread Stack allocator
template<typename size_type>
using StackAddressAllocatorST = StackAddressAllocator<size_type>;
**/

}
}

#include "irr/core/alloc/AddressAllocatorConcurrencyAdaptors.h"
namespace irr
{
namespace core
{

//

}
}

#endif // __IRR_STACK_ADDRESS_ALLOCATOR_H_INCLUDED__

