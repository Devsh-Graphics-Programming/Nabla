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

    template<class ReadOnlyDataAccessor, class OutputAccessor, class StatusAccessor, class ScratchAccessor>
    static void __call(NBL_REF_ARG(ReadOnlyDataAccessor) dataAccessor, NBL_REF_ARG(OutputAccessor) outputAccessor, NBL_REF_ARG(StatusAccessor) statusAccessor, NBL_REF_ARG(ScratchAccessor) sharedMemScratchAccessor)
    {
        impl::reduce<Config, BinOp, ForwardProgressGuarantees, device_capabilities> fn;
        fn.template __call<ReadOnlyDataAccessor,OutputAccessor,StatusAccessor,ScratchAccessor>(dataAccessor, outputAccessor, statusAccessor, sharedMemScratchAccessor);
    }
};

}
}
}

#endif
