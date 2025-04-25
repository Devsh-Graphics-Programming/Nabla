// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP2_ARITHMETIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP2_ARITHMETIC_INCLUDED_


#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/workgroup/ballot.hlsl"
#include "nbl/builtin/hlsl/workgroup/broadcast.hlsl"
#include "nbl/builtin/hlsl/workgroup2/shared_scan.hlsl"


namespace nbl
{
namespace hlsl
{
namespace workgroup2
{



template<class BinOp, uint16_t ItemCount, uint16_t ElementsPerInvocation, class device_capabilities=void>
struct reduction
{
    using scalar_t = typename BinOp::type_t;

    template<class Accessor>
    static scalar_t __call(type_t input[ElementsPerInvocation], NBL_REF_ARG(Accessor) accessor)[ElementsPerInvocation]
    {
        impl::reduce<BinOp,ItemCount,device_capabilities> fn;
        return fn.output;
    }
}

}
}
}

#endif
