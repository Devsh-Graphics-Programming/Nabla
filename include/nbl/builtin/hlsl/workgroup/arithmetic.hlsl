// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_ARITHMETIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_ARITHMETIC_INCLUDED_

#include "nbl/builtin/hlsl/workgroup/ballot.hlsl"
#include "nbl/builtin/hlsl/workgroup/broadcast.hlsl"
#include "nbl/builtin/hlsl/workgroup/shared_scan.hlsl"

namespace nbl
{
namespace hlsl
{
namespace workgroup
{

template<typename T, class Binop, class SharedAccessor>
struct reduction
{
    struct inclusive_scan_t : subgroup::inclusive_scan<T, Binop> {}; // Yes, inclusive scan subgroup op is used for reduction workgroup ops
    T operator()(T value)
    {
        SharedAccessor accessor;
        accessor.main.workgroupExecutionAndMemoryBarrier();
        WorkgroupScanHead<T, inclusive_scan_t, SharedAccessor> wsh = WorkgroupScanHead<T, inclusive_scan_t, SharedAccessor>::create(false, Binop::identity(), _NBL_HLSL_WORKGROUP_SIZE_);
        T result = wsh(value);
        accessor.main.workgroupExecutionAndMemoryBarrier();
        T retVal = Broadcast<uint, SharedAccessor>(result, wsh.lastInvocationInLevel);
        accessor.main.workgroupExecutionAndMemoryBarrier();
        return retVal;
    }
};

template<typename T, class Binop, class SharedAccessor>
struct inclusive_scan
{
    struct inclusive_scan_t : subgroup::inclusive_scan<T, Binop> {};
    T operator()(T value)
    {
        SharedAccessor accessor;
        accessor.main.workgroupExecutionAndMemoryBarrier();
        WorkgroupScanHead<T, inclusive_scan_t, SharedAccessor> wsh = WorkgroupScanHead<T, inclusive_scan_t, SharedAccessor>::create(true, Binop::identity(), _NBL_HLSL_WORKGROUP_SIZE_);
        wsh(value);
        WorkgroupScanTail<T, Binop, SharedAccessor> wst = WorkgroupScanTail<T, Binop, SharedAccessor>::create(false, Binop::identity(), wsh.firstLevelScan, wsh.lastInvocation, wsh.scanStoreIndex);
        T retVal = wst();
        accessor.main.workgroupExecutionAndMemoryBarrier();
        return retVal;
    }
};

template<typename T, class Binop, class SharedAccessor>
struct exclusive_scan
{
    struct inclusive_scan_t : subgroup::inclusive_scan<T, Binop> {}; // Yes, inclusive scan subgroup op is used for exclusive workgroup ops
    T operator()(T value)
    {
        SharedAccessor accessor;
        accessor.main.workgroupExecutionAndMemoryBarrier();
        WorkgroupScanHead<T, inclusive_scan_t, SharedAccessor> wsh = WorkgroupScanHead<T, inclusive_scan_t, SharedAccessor>::create(true, Binop::identity(), _NBL_HLSL_WORKGROUP_SIZE_);
        wsh(value);
        WorkgroupScanTail<T, Binop, SharedAccessor> wst = WorkgroupScanTail<T, Binop, SharedAccessor>::create(true, Binop::identity(), wsh.firstLevelScan, wsh.lastInvocation, wsh.scanStoreIndex);
        T retVal = wst();
        accessor.main.workgroupExecutionAndMemoryBarrier();
        return retVal;
    }
};

#define WSHT WorkgroupScanHead<uint, subgroup::inclusive_scan<uint, binops::add<uint> >, SharedAccessor>
#define WSTT WorkgroupScanTail<uint, binops::add<uint>, SharedAccessor>
template<class SharedAccessor>
uint ballotScanBitCount(in bool exclusive)
{
    SharedAccessor accessor;
    const uint _dword = getDWORD(gl_LocalInvocationIndex);
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
        
        WSHT wsh = WSHT::create(true, 0u, uballotBitfieldCount);
        wsh(countbits(localBitfieldBackup));
        
        WSTT wst = WSTT::create(true, 0u, wsh.firstLevelScan, wsh.lastInvocation, wsh.scanStoreIndex);
        uint bitscan = wst();
        
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
uint ballotInclusiveBitCount()
{
    return ballotScanBitCount<SharedAccessor>(false);
}

template<class SharedAccessor>
uint ballotExclusiveBitCount()
{
    return ballotScanBitCount<SharedAccessor>(true);
}

#undef WSHT
#undef WSTT

}
}
}

#endif