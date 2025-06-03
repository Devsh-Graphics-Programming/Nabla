// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SCAN_ARITHMETIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_SCAN_ARITHMETIC_INCLUDED_

#include "nbl/builtin/hlsl/scan/arithmetic_impl.hlsl"

namespace nbl
{
namespace hlsl
{
namespace scan
{

template<class Config, class BinOp, bool ForwardProgressGuarantees, class device_capabilities=void>
struct reduction
{
    using scalar_t = typename BinOp::type_t;

    template<class ReadOnlyDataAccessor, class ScratchAccessor NBL_FUNC_REQUIRES(ArithmeticDataAccessor<ReadOnlyDataAccessor> && ArithmeticSharedMemoryAccessor<ScratchAccessor>)
    static scalar_t __call(NBL_REF_ARG(ReadOnlyDataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) sharedMemScratchAccessor)   // scratch bda?
    {
        impl::reduce<Config, BinOp, ForwardProgressGuarantees, device_capabilities> fn;
        scalar_t value = fn.template __call<ReadOnlyDataAccessor,ScratchAccessor>(dataAccessor, sharedMemScratchAccessor);
        return value;
    }
};

}
}
}

#endif
