// Copyright (C) 2018 Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW Engine"
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_ADDRESS_ALLOCATOR_STATE_TYPES_H_INCLUDED__
#define __IRR_ADDRESS_ALLOCATOR_STATE_TYPES_H_INCLUDED__

#include "IrrCompileConfig.h"

#include "irr/core/alloc/AddressAllocatorState.h"

#include "irr/core/alloc/LinearAddressAllocator.h"
#include "irr/core/alloc/PoolAddressAllocator.h"

namespace irr
{
namespace core
{

//! Regular states

// linear
template<typename AddressType>
using LinearAddressAllocatorStateST = AddressAllocatorState<LinearAddressAllocatorST<AddressType> >;
template<typename AddressType, class BasicLockable>
using LinearAddressAllocatorStateMT = AddressAllocatorState<LinearAddressAllocatorMT<AddressType,BasicLockable> >;
// pool
template<typename AddressType>
using PoolAddressAllocatorStateST = AddressAllocatorState<PoolAddressAllocatorST<AddressType> >;
template<typename AddressType, class BasicLockable>
using PoolAddressAllocatorStateMT = AddressAllocatorState<PoolAddressAllocatorMT<AddressType,BasicLockable> >;



//! Buffer States

// linear
template<class I, typename AddressType>
using LinearAddressAllocatorRefCountedStateST = AllocatorStateRefCountedAdaptor<I, LinearAddressAllocatorST<AddressType> >;
template<class I, typename AddressType, class BasicLockable>
using LinearAddressAllocatorRefCountedStateMT = AllocatorStateRefCountedAdaptor<I, LinearAddressAllocatorMT<AddressType,BasicLockable> >;
// pool
template<class I, typename AddressType>
using PoolAddressAllocatorRefCountedStateST = AllocatorStateRefCountedAdaptor<I, PoolAddressAllocatorST<AddressType> >;
template<class I, typename AddressType, class BasicLockable>
using PoolAddressAllocatorRefCountedStateMT = AllocatorStateRefCountedAdaptor<I, PoolAddressAllocatorMT<AddressType,BasicLockable> >;

}
}

#endif // __IRR_ADDRESS_ALLOCATOR_STATE_TYPES_H_INCLUDED__
