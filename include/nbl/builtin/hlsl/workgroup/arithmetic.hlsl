// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_ARITHMETIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_ARITHMETIC_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/workgroup/ballot.hlsl"
#include "nbl/builtin/hlsl/workgroup/broadcast.hlsl"
#include "nbl/builtin/hlsl/workgroup/shared_scan.hlsl"

namespace nbl
{
namespace hlsl
{
namespace workgroup
{

#define REDUCE Reduce<T, subgroup::inclusive_scan<T, Binop>, SharedAccessor, _NBL_HLSL_WORKGROUP_SIZE_>
#define SCAN(isExclusive) Scan<T, Binop, subgroup::inclusive_scan<T, Binop>, SharedAccessor, _NBL_HLSL_WORKGROUP_SIZE_, isExclusive>
template<typename T, class Binop, class SharedAccessor>
T reduction(T value, NBL_REF_ARG(SharedAccessor) accessor)
{
    REDUCE reduce = REDUCE::create();
    reduce(value, accessor);
    accessor.main.workgroupExecutionAndMemoryBarrier();
    T retVal = Broadcast<uint, SharedAccessor>(reduce.lastLevelScan, accessor, reduce.lastInvocationInLevel);
    return retVal;
}

template<typename T, class Binop, class SharedAccessor>
T inclusive_scan(T value, NBL_REF_ARG(SharedAccessor) accessor)
{
    SCAN(false) incl_scan = SCAN(false)::create();
    T retVal = incl_scan(value, accessor);
    return retVal;
}

template<typename T, class Binop, class SharedAccessor>
T exclusive_scan(T value, NBL_REF_ARG(SharedAccessor) accessor)
{
    SCAN(true) excl_scan = SCAN(true)::create();
    T retVal = excl_scan(value, accessor);
    return retVal;
}

#undef REDUCE
#undef SCAN

#define REDUCE Reduce<uint, subgroup::inclusive_scan<uint, plus<uint> >, SharedAccessor, impl::uballotBitfieldCount>
#define SCAN Scan<uint, plus<uint>, subgroup::inclusive_scan<uint, plus<uint> >, SharedAccessor, impl::uballotBitfieldCount, true>
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
    uint participatingBitfield = 0;
    if(SubgroupContiguousIndex() < impl::uballotBitfieldCount)
    {
        participatingBitfield = accessor.ballot.get(SubgroupContiguousIndex());
    }
    accessor.ballot.workgroupExecutionAndMemoryBarrier();
    REDUCE reduce = REDUCE::create();
    reduce(countbits(participatingBitfield), accessor);
    accessor.main.workgroupExecutionAndMemoryBarrier();
    return Broadcast<uint, SharedAccessor>(reduce.lastLevelScan, accessor, reduce.lastInvocationInLevel);
}

template<class SharedAccessor>
uint ballotScanBitCount(const bool exclusive, NBL_REF_ARG(SharedAccessor) accessor)
{
    const uint _dword = impl::getDWORD(SubgroupContiguousIndex());
    const uint localBitfield = accessor.ballot.get(_dword);
    uint globalCount;
    {
        uint participatingBitfield;
        if(SubgroupContiguousIndex() < impl::uballotBitfieldCount)
        {
            participatingBitfield = accessor.ballot.get(SubgroupContiguousIndex());
        }
        // scan hierarchically, invocations with `SubgroupContiguousIndex() >= uballotBitfieldCount` will have garbage here
        accessor.ballot.workgroupExecutionAndMemoryBarrier();
        
        SCAN scan = SCAN::create();
        uint bitscan = scan(countbits(participatingBitfield), accessor);
        
        accessor.main.set(SubgroupContiguousIndex(), bitscan);
        accessor.main.workgroupExecutionAndMemoryBarrier();
        
        // fix it (abuse the fact memory is left over)
        globalCount = _dword != 0u ? accessor.main.get(_dword) : 0u;
        accessor.main.workgroupExecutionAndMemoryBarrier();
    }
    const uint mask = (exclusive ? 0x7fFFffFFu:0xFFffFFffu)>>(31u-(SubgroupContiguousIndex()&31u));
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

#undef REDUCE
#undef SCAN

}
}
}

#endif