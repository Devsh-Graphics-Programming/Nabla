// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_BROADCAST_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_BROADCAST_INCLUDED_

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
template<typename T, class SharedAccessor, bool edgeBarriers = true>
T Broadcast(in T val, in uint id)
{
    SharedAccessor accessor;
    
    if(edgeBarriers)
        accessor.broadcast.workgroupExecutionAndMemoryBarrier();
    
    if(gl_LocalInvocationIndex == id) {
        accessor.broadcast.set(uballotBitfieldCount, val);
    }
    
    if(edgeBarriers)
        accessor.broadcast.workgroupExecutionAndMemoryBarrier();
    
    return accessor.broadcast.get(uballotBitfieldCount);
}

template<typename T, class SharedAccessor, bool edgeBarriers = true>
T BroadcastFirst(in T val)
{
    SharedAccessor accessor;
    
    if(edgeBarriers)
        accessor.broadcast.workgroupExecutionAndMemoryBarrier();
    
    if (Elect())
        accessor.broadcast.set(uballotBitfieldCount, val);
    
    if(edgeBarriers)
        accessor.broadcast.workgroupExecutionAndMemoryBarrier();
    
    return accessor.broadcast.get(uballotBitfieldCount);
}

}
}
}
#endif