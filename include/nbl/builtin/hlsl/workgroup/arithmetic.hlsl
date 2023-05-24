// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_ARITHMETIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_ARITHMETIC_INCLUDED_

#include "nbl/builtin/hlsl/workgroup/ballot.hlsl"

#ifndef _NBL_GL_LOCAL_INVOCATION_IDX_DECLARED_
#define _NBL_GL_LOCAL_INVOCATION_IDX_DECLARED_
const uint gl_LocalInvocationIndex : SV_GroupIndex;
#endif

namespace nbl
{
namespace hlsl
{
namespace workgroup
{

template<typename T, class Binop, class ScratchAccessor>
struct reduction
{
    struct reduction_t : subgroup::reduction<T, Binop, ScratchAccessor> {};
    T operator()(T value)
    {
        WorkgroupScanHead<T, reduction_t, ScratchAccessor> wsh = WorkgroupScanHead<T, reduction_t, ScratchAccessor>::create(false, 0xffFFffFFu, _NBL_HLSL_WORKGROUP_SIZE_);
        T result = wsh(value);
        Barrier();
		// REVIEW: I think broadcast is fine to use the ScratchAccessor at this point since reduction has finished
        return broadcast<uint, ScratchAccessor>(result, wsh.lastInvocationInLevel);
    }
};

template<typename T, class Binop, class ScratchAccessor>
struct exclusive_scan
{
    struct exclusive_scan_t : subgroup::exclusive_scan<T, Binop, ScratchAccessor> {};
    T operator()(T value)
    {
        WorkgroupScanHead<T, exclusive_scan_t, ScratchAccessor> wsh = WorkgroupScanHead<T, exclusive_scan_t, ScratchAccessor>::create(true, 0xffFFffFFu, _NBL_HLSL_WORKGROUP_SIZE_);
        wsh(value);
        WorkgroupScanTail<T, Binop, ScratchAccessor> wst = WorkgroupScanTail<T, Binop, ScratchAccessor>::create(true, 0xffFFffFFu, wsh.firstLevelScan, wsh.lastInvocation, wsh.scanStoreIndex);
		return wst();
    }
};

template<typename T, class Binop, class ScratchAccessor>
struct inclusive_scan
{
    struct inclusive_scan_t : subgroup::inclusive_scan<T, Binop, ScratchAccessor> {};
    T operator()(T value)
    {
        WorkgroupScanHead<T, inclusive_scan_t, ScratchAccessor> wsh = WorkgroupScanHead<T, inclusive_scan_t, ScratchAccessor>::create(true, 0xffFFffFFu, _NBL_HLSL_WORKGROUP_SIZE_);
        wsh(value);
        WorkgroupScanTail<T, Binop, ScratchAccessor> wst = WorkgroupScanTail<T, Binop, ScratchAccessor>::create(false, 0xffFFffFFu, wsh.firstLevelScan, wsh.lastInvocation, wsh.scanStoreIndex);
		return wst();
    }
};

}
}
}

#endif