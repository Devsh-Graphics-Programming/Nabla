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

template<typename T, class Binop, class ScratchAccessor>
struct reduction
{
    struct reduction_t : subgroup::reduction<T, Binop, ScratchAccessor, false> {};
    T operator()(T value)
    {
		Binop op;
        WorkgroupScanHead<T, reduction_t, ScratchAccessor> wsh = WorkgroupScanHead<T, reduction_t, ScratchAccessor>::create(false, op.identity(), _NBL_HLSL_WORKGROUP_SIZE_);
        T result = wsh(value);
        Barrier();
        return Broadcast<uint, ScratchAccessor>(result, wsh.lastInvocationInLevel);
    }
};

template<typename T, class Binop, class ScratchAccessor>
struct inclusive_scan
{
    struct inclusive_scan_t : subgroup::inclusive_scan<T, Binop, ScratchAccessor, false> {};
    T operator()(T value)
    {
		Binop op;
        WorkgroupScanHead<T, inclusive_scan_t, ScratchAccessor> wsh = WorkgroupScanHead<T, inclusive_scan_t, ScratchAccessor>::create(true, op.identity(), _NBL_HLSL_WORKGROUP_SIZE_);
        wsh(value);
        WorkgroupScanTail<T, Binop, ScratchAccessor> wst = WorkgroupScanTail<T, Binop, ScratchAccessor>::create(false, op.identity(), wsh.firstLevelScan, wsh.lastInvocation, wsh.scanStoreIndex);
		return wst();
    }
};

template<typename T, class Binop, class ScratchAccessor>
struct exclusive_scan
{
    struct inclusive_scan_t : subgroup::inclusive_scan<T, Binop, ScratchAccessor, false> {}; // Yes, inclusive scan subgroup op is used for exclusive workgroup ops
    T operator()(T value)
    {
		Binop op;
        WorkgroupScanHead<T, inclusive_scan_t, ScratchAccessor> wsh = WorkgroupScanHead<T, inclusive_scan_t, ScratchAccessor>::create(true, op.identity(), _NBL_HLSL_WORKGROUP_SIZE_);
        wsh(value);
        WorkgroupScanTail<T, Binop, ScratchAccessor> wst = WorkgroupScanTail<T, Binop, ScratchAccessor>::create(true, op.identity(), wsh.firstLevelScan, wsh.lastInvocation, wsh.scanStoreIndex);
		return wst();
    }
};

#define WSHT WorkgroupScanHead<uint, subgroup::inclusive_scan<uint, binops::add<uint>, ScratchAccessor>, ScratchAccessor>
#define WSTT WorkgroupScanTail<uint, binops::add<uint>, ScratchAccessor>
template<class ScratchAccessor>
uint ballotScanBitCount(in bool exclusive)
{
	ScratchAccessor scratch;
	const uint _dword = getDWORD(gl_LocalInvocationIndex);
	const uint localBitfield = scratch.main.get(_dword);
	uint globalCount;
	{
		uint localBitfieldBackup;
		if(gl_LocalInvocationIndex < bitfieldDWORDs)
		{
			localBitfieldBackup = scratch.main.get(gl_LocalInvocationIndex);
		}
		// scan hierarchically, invocations with `gl_LocalInvocationIndex >= bitfieldDWORDs` will have garbage here
		Barrier();
		
		WSHT wsh = WSHT::create(true, 0u, bitfieldDWORDs);
		wsh(countbits(localBitfieldBackup));
		
		WSTT wst = WSTT::create(true, 0u, wsh.firstLevelScan, wsh.lastInvocation, wsh.scanStoreIndex);
		wst();
		
		// fix it (abuse the fact memory is left over)
		globalCount = _dword != 0u ? scratch.main.get(_dword) : 0u;
		Barrier();
		
		// restore
		if(gl_LocalInvocationIndex < bitfieldDWORDs)
		{
			scratch.main.set(gl_LocalInvocationIndex, localBitfieldBackup);
		}
		Barrier();
	}
	const uint mask = (exclusive ? 0x7fFFffFFu:0xFFffFFffu)>>(31u-(gl_LocalInvocationIndex&31u));
	return globalCount + countbits(localBitfield & mask);
}

template<class ScratchAccessor>
uint ballotInclusiveBitCount()
{
	return ballotScanBitCount<ScratchAccessor>(false);
}

template<class ScratchAccessor>
uint ballotExclusiveBitCount()
{
	return ballotScanBitCount<ScratchAccessor>(true);
}

}
}
}

#endif