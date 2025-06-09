// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP2_ARITHMETIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP2_ARITHMETIC_INCLUDED_

#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/concepts/accessors/workgroup_arithmetic.hlsl"
#include "nbl/builtin/hlsl/workgroup2/shared_scan.hlsl"

namespace nbl
{
namespace hlsl
{
namespace workgroup2
{

template<class Config, class BinOp, class device_capabilities=void NBL_PRIMARY_REQUIRES(is_configuration_v<Config>)
struct reduction
{
    using scalar_t = typename BinOp::type_t;

    template<class ReadOnlyDataAccessor, class ScratchAccessor NBL_FUNC_REQUIRES(ArithmeticReadOnlyDataAccessor<ReadOnlyDataAccessor,scalar_t> && ArithmeticSharedMemoryAccessor<ScratchAccessor,scalar_t>)
    static scalar_t __call(NBL_REF_ARG(ReadOnlyDataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor)
    {
        impl::reduce<Config,BinOp,Config::LevelCount,device_capabilities> fn;
        return fn.template __call<ReadOnlyDataAccessor,ScratchAccessor>(dataAccessor, scratchAccessor);
    }
};

template<class Config, class BinOp, class device_capabilities=void NBL_PRIMARY_REQUIRES(is_configuration_v<Config>)
struct inclusive_scan
{
    using scalar_t = typename BinOp::type_t;

    template<class DataAccessor, class ScratchAccessor NBL_FUNC_REQUIRES(ArithmeticDataAccessor<DataAccessor,scalar_t> && ArithmeticSharedMemoryAccessor<ScratchAccessor,scalar_t>)
    static void __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor)
    {
        impl::scan<Config,BinOp,false,Config::LevelCount,device_capabilities> fn;
        fn.template __call<DataAccessor,ScratchAccessor>(dataAccessor, scratchAccessor);
    }
};

template<class Config, class BinOp, class device_capabilities=void NBL_PRIMARY_REQUIRES(is_configuration_v<Config>)
struct exclusive_scan
{
    using scalar_t = typename BinOp::type_t;

    template<class DataAccessor, class ScratchAccessor NBL_FUNC_REQUIRES(ArithmeticDataAccessor<DataAccessor,scalar_t> && ArithmeticSharedMemoryAccessor<ScratchAccessor,scalar_t>)
    static void __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor)
    {
        impl::scan<Config,BinOp,true,Config::LevelCount,device_capabilities> fn;
        fn.template __call<DataAccessor,ScratchAccessor>(dataAccessor, scratchAccessor);
    }
};

}
}
}

#endif
