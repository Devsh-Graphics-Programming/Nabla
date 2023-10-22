// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_BROADCAST_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_BROADCAST_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/workgroup/ballot.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace workgroup
{

/**
 * Broadcasts the value `val` of invocation index `id`
 * to all other invocations.
 * 
 * We save the value in the shared array in the uballotBitfieldCount index 
 * and then all invocations access that index.
 */
template<typename T, class SharedAccessor>
T Broadcast(const T val, NBL_REF_ARG(SharedAccessor) accessor, const uint id)
{
    if(SubgroupContiguousIndex() == id) {
        accessor.broadcast.set(impl::uballotBitfieldCount, val);
    }
    
    accessor.broadcast.workgroupExecutionAndMemoryBarrier();
    
    return accessor.broadcast.get(impl::uballotBitfieldCount);
}

template<typename T, class SharedAccessor>
T BroadcastFirst(const T val, NBL_REF_ARG(SharedAccessor) accessor)
{
    if (Elect())
        accessor.broadcast.set(impl::uballotBitfieldCount, val);
    
    accessor.broadcast.workgroupExecutionAndMemoryBarrier();
    
    return accessor.broadcast.get(impl::uballotBitfieldCount);
}

}
}
}
#endif