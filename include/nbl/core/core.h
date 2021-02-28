// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_H_INCLUDED__
#define __NBL_CORE_H_INCLUDED__

#include "nbl/core/compile_config.h"

//overarching includes (compiler tricks mostly)
#include "nbl/macros.h"
#include "nbl/type_traits.h"
// allocator
#include "nbl/core/alloc/AddressAllocatorBase.h"
#include "nbl/core/alloc/AddressAllocatorConcurrencyAdaptors.h"
#include "nbl/core/alloc/address_allocator_traits.h"
#include "nbl/core/alloc/AlignedBase.h"
#include "nbl/core/alloc/aligned_allocator.h"
#include "nbl/core/alloc/aligned_allocator_adaptor.h"
#include "nbl/core/alloc/AllocatorTrivialBases.h"
#include "nbl/core/alloc/GeneralpurposeAddressAllocator.h"
#include "nbl/core/alloc/HeterogenousMemoryAddressAllocatorAdaptor.h"
#include "nbl/core/alloc/IAddressAllocator.h"
#include "nbl/core/alloc/IAllocator.h"
#include "nbl/core/alloc/LinearAddressAllocator.h"
#include "nbl/core/alloc/MultiBufferingAllocatorBase.h"
#include "nbl/core/alloc/null_allocator.h"
#include "nbl/core/alloc/PoolAddressAllocator.h"
#include "nbl/core/alloc/ResizableHeterogenousMemoryAllocator.h"
#include "nbl/core/alloc/StackAddressAllocator.h"
#include "nbl/core/alloc/SimpleBlockBasedAllocator.h"
// algorithm
#include "nbl/core/algorithm/radix_sort.h"
// containers
#include "nbl/core/containers/dynamic_array.h"
#include "nbl/core/containers/refctd_dynamic_array.h"
#include "nbl/core/containers/FixedCapacityDoublyLinkedList.h"
#include "nbl/core/containers/LRUCache.h"
// math
#include "nbl/core/math/intutil.h"
#include "nbl/core/math/floatutil.tcc"
#include "nbl/core/math/colorutil.h"
#include "nbl/core/math/glslFunctions.tcc"
#include "nbl/core/math/rational.h"
#include "nbl/core/math/plane3dSIMD.h"
// memory
#include "nbl/core/memory/memory.h"
#include "nbl/core/memory/new_delete.h"
#include "nbl/core/memory/CLeakDebugger.h"
// samplers
#include "nbl/core/sampling/RandomSampler.h"
#include "nbl/core/sampling/SobolSampler.h"
#include "nbl/core/sampling/OwenSampler.h"
// parallel
#include "nbl/core/parallel/IThreadBound.h"
#include "nbl/core/parallel/unlock_guard.h"
// string
#include "nbl/core/string/stringutil.h"
#include "nbl/core/string/UniqueStringLiteralType.h"
// other useful things
#include "nbl/core/BaseClasses.h"
#include "nbl/core/SingleEventHandler.h"
#include "nbl/core/EventDeferredHandler.h"
#include "nbl/core/IBuffer.h"
#include "nbl/core/IReferenceCounted.h"
#include "nbl/core/SRAIIBasedExiter.h"
#include "nbl/core/SRange.h"
#include "nbl/core/Types.h"

// implementations
#include "matrix3x4SIMD_impl.h"
#include "matrix4SIMD_impl.h"

#endif