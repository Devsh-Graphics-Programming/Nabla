// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_ARITHMETIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_ARITHMETIC_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat/cpp_compat.h"
#include "nbl/builtin/hlsl/workgroup/ballot.hlsl"
#include "nbl/builtin/hlsl/workgroup/broadcast.hlsl"
#include "nbl/builtin/hlsl/workgroup/shared_scan.hlsl"

namespace nbl
{
namespace hlsl
{
namespace workgroup
{

#define WSHT WorkgroupScanHead<T, subgroup::inclusive_scan<T, Binop>, SharedAccessor>
#define WSTT(isExclusive) WorkgroupScanTail<T, Binop, WSHT, SharedAccessor, isExclusive>
template<typename T, class Binop, class SharedAccessor>
T reduction(T value, NBL_REF_ARG(SharedAccessor) accessor)
{
    accessor.main.workgroupExecutionAndMemoryBarrier();
    WSHT wsh = WSHT::create(Binop::identity(), _NBL_HLSL_WORKGROUP_SIZE_);
    wsh(value, accessor);
    accessor.main.workgroupExecutionAndMemoryBarrier();
    T retVal = Broadcast<uint, SharedAccessor>(wsh.lastLevelScan, accessor, wsh.lastInvocationInLevel);
    accessor.main.workgroupExecutionAndMemoryBarrier();
    return retVal;
}

template<typename T, class Binop, class SharedAccessor>
T inclusive_scan(T value, NBL_REF_ARG(SharedAccessor) accessor)
{
    accessor.main.workgroupExecutionAndMemoryBarrier();
    WSHT wsh = WSHT::create(Binop::identity(), _NBL_HLSL_WORKGROUP_SIZE_);
    wsh(value, accessor);
    WSTT(false) incl_wst = WSTT(false)::create(Binop::identity(), wsh);
    T retVal = incl_wst(accessor);
    accessor.main.workgroupExecutionAndMemoryBarrier();
    return retVal;
}

template<typename T, class Binop, class SharedAccessor>
T exclusive_scan(T value, NBL_REF_ARG(SharedAccessor) accessor)
{
    accessor.main.workgroupExecutionAndMemoryBarrier();
    WSHT wsh = WSHT::create(Binop::identity(), _NBL_HLSL_WORKGROUP_SIZE_);
    wsh(value, accessor);
    WSTT(true) excl_wst = WSTT(true)::create(Binop::identity(), wsh);
    T retVal = excl_wst(accessor);
    accessor.main.workgroupExecutionAndMemoryBarrier();
    return retVal;
}

#undef WSHT
#undef WSTT

/**
 * Gives us the sum (reduction) of all ballots for the workgroup.
 *
 * Only the first few invocations are used for performing the sum 
 * since we only have `uballotBitfieldCount` amount of uints that we need 
 * to add together.
 * 
 * We add them all in the shared array index after the last DWORD 
 * that is used for the ballots. For example, if we have 128 workgroup size,
 * then the array index in which we accumulate the sum is `4` since 
 * indexes 0..3 are used for ballots.
 */ 
template<class SharedAccessor>
uint ballotBitCount(NBL_REF_ARG(SharedAccessor) accessor)
{
    accessor.main.set(uballotBitfieldCount, 0u);
    accessor.main.workgroupExecutionAndMemoryBarrier();
    if(gl_LocalInvocationIndex < uballotBitfieldCount)
    {
        const uint localBallot = accessor.main.get(gl_LocalInvocationIndex);
        const uint localBallotBitCount = countbits(localBallot);
        uint dummy;
        accessor.main.atomicAdd(uballotBitfieldCount, localBallotBitCount, dummy);
    }
    accessor.main.workgroupExecutionAndMemoryBarrier();
    return accessor.main.get(uballotBitfieldCount);
}

#define WSHT WorkgroupScanHead<uint, subgroup::inclusive_scan<uint, binops::add<uint> >, SharedAccessor>
#define WSTT WorkgroupScanTail<uint, binops::add<uint>, WSHT, SharedAccessor, true>
template<class SharedAccessor>
uint ballotScanBitCount(in bool exclusive, NBL_REF_ARG(SharedAccessor) accessor)
{
    const uint _dword = impl::getDWORD(gl_LocalInvocationIndex);
    const uint localBitfield = accessor.main.get(_dword);
    uint globalCount;
    {
        uint localBitfieldBackup;
        if(gl_LocalInvocationIndex < uballotBitfieldCount)
        {
            localBitfieldBackup = accessor.main.get(gl_LocalInvocationIndex);
        }
        // scan hierarchically, invocations with `gl_LocalInvocationIndex >= uballotBitfieldCount` will have garbage here
        accessor.main.workgroupExecutionAndMemoryBarrier();
        
        WSHT wsh = WSHT::create(0u, uballotBitfieldCount);
        wsh(countbits(localBitfieldBackup), accessor);
        
        WSTT wst = WSTT::create(0u, wsh);
        uint bitscan = wst(accessor);
        
        accessor.main.set(gl_LocalInvocationIndex, bitscan);
        accessor.main.workgroupExecutionAndMemoryBarrier();
        
        // fix it (abuse the fact memory is left over)
        globalCount = _dword != 0u ? accessor.main.get(_dword) : 0u;
        accessor.main.workgroupExecutionAndMemoryBarrier();
        
        // restore because the counting process has changed the ballots in the shared mem
        // and we might want to use them further
        if(gl_LocalInvocationIndex < uballotBitfieldCount)
        {
            accessor.main.set(gl_LocalInvocationIndex, localBitfieldBackup);
        }
        accessor.main.workgroupExecutionAndMemoryBarrier();
    }
    const uint mask = (exclusive ? 0x7fFFffFFu:0xFFffFFffu)>>(31u-(gl_LocalInvocationIndex&31u));
    return globalCount + countbits(localBitfield & mask);
}

template<class SharedAccessor>
uint ballotInclusiveBitCount(NBL_REF_ARG(SharedAccessor) accessor)
{
    return ballotScanBitCount<SharedAccessor>(false, accessor);
}

template<class SharedAccessor>
uint ballotExclusiveBitCount(NBL_REF_ARG(SharedAccessor) accessor)
{
    return ballotScanBitCount<SharedAccessor>(true, accessor);
}

#undef WSHT
#undef WSTT

}
}
}

#endif