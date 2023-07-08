// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_ARITHMETIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_ARITHMETIC_INCLUDED_

#include "nbl/builtin/hlsl/workgroup/ballot.hlsl"

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
		// REVIEW: I think broadcast is fine to use the ScratchAccessor at this point since reduction has finished
        return broadcast<uint, ScratchAccessor>(result, wsh.lastInvocationInLevel);
    }
};

template<typename T, class Binop, class ScratchAccessor>
struct exclusive_scan
{
    struct exclusive_scan_t : subgroup::exclusive_scan<T, Binop, ScratchAccessor, false> {};
    T operator()(T value)
    {
		Binop op;
        WorkgroupScanHead<T, exclusive_scan_t, ScratchAccessor> wsh = WorkgroupScanHead<T, exclusive_scan_t, ScratchAccessor>::create(true, op.identity(), _NBL_HLSL_WORKGROUP_SIZE_);
        wsh(value);
        WorkgroupScanTail<T, Binop, ScratchAccessor> wst = WorkgroupScanTail<T, Binop, ScratchAccessor>::create(true, op.identity(), wsh.firstLevelScan, wsh.lastInvocation, wsh.scanStoreIndex);
		return wst();
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

}
}
}

#endif