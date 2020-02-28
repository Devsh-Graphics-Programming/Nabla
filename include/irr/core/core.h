#ifndef __IRR_CORE_H_INCLUDED__
#define __IRR_CORE_H_INCLUDED__

//overarching includes (compiler tricks mostly)
#include "IrrCompileConfig.h" // what's this still doing here?
#include "irr/macros.h"
#include "irr/static_if.h"
#include "irr/switch_constexpr.h"
#include "irr/type_traits.h"
#include "irr/void_t.h"
// allocator
#include "irr/core/alloc/AddressAllocatorBase.h"
#include "irr/core/alloc/AddressAllocatorConcurrencyAdaptors.h"
#include "irr/core/alloc/address_allocator_traits.h"
#include "irr/core/alloc/AlignedBase.h"
#include "irr/core/alloc/aligned_allocator.h"
#include "irr/core/alloc/aligned_allocator_adaptor.h"
#include "irr/core/alloc/AllocatorTrivialBases.h"
#include "irr/core/alloc/ContiguousPoolAddressAllocator.h"
#include "irr/core/alloc/GeneralpurposeAddressAllocator.h"
#include "irr/core/alloc/HeterogenousMemoryAddressAllocatorAdaptor.h"
#include "irr/core/alloc/IAddressAllocator.h"
#include "irr/core/alloc/IAllocator.h"
#include "irr/core/alloc/LinearAddressAllocator.h"
#include "irr/core/alloc/MultiBufferingAllocatorBase.h"
#include "irr/core/alloc/null_allocator.h"
#include "irr/core/alloc/PoolAddressAllocator.h"
#include "irr/core/alloc/ResizableHeterogenousMemoryAllocator.h"
#include "irr/core/alloc/StackAddressAllocator.h"
#include "irr/core/alloc/SimpleBlockBasedAllocator.h"
// math
#include "irr/core/math/intutil.h"
#include "irr/core/math/floatutil.tcc"
#include "irr/core/math/glslFunctions.tcc"
#include "irr/core/math/rational.h"
#include "irr/core/math/plane3dSIMD.h"
// memory
#include "irr/core/memory/memory.h"
#include "irr/core/memory/new_delete.h"
#include "irr/core/memory/dynamic_array.h"
#include "irr/core/memory/refctd_dynamic_array.h"
#include "irr/core/memory/CLeakDebugger.h"
// samplers
#include "irr/core/sampling/RandomSampler.h"
#include "irr/core/sampling/SobolSampler.h"
#include "irr/core/sampling/OwenSampler.h"
// parallel
#include "irr/core/parallel/IThreadBound.h"
#include "irr/core/parallel/unlock_guard.h"
// string
#include "irr/core/string/stringutil.h"
// other useful things
#include "irr/core/BaseClasses.h"
#include "irr/core/EventDeferredHandler.h"
#include "irr/core/IBuffer.h"
#include "irr/core/IReferenceCounted.h"
#include "irr/core/SRAIIBasedExiter.h"
#include "irr/core/SRange.h"
#include "irr/core/Types.h"

// implementations
#include "matrix3x4SIMD_impl.h"
#include "matrix4SIMD_impl.h"

#endif