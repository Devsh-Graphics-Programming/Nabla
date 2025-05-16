// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP2_ARITHMETIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP2_ARITHMETIC_INCLUDED_


#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/workgroup/ballot.hlsl"
#include "nbl/builtin/hlsl/workgroup/broadcast.hlsl"
#include "nbl/builtin/hlsl/concepts/accessors/workgroup_arithmetic.hlsl"
#include "nbl/builtin/hlsl/workgroup2/shared_scan.hlsl"


namespace nbl
{
namespace hlsl
{
namespace workgroup2
{

template<class Config, class BinOp, class device_capabilities=void>
struct reduction
{
    template<class DataAccessor, class ScratchAccessor NBL_FUNC_REQUIRES(ArithmeticDataAccessor<DataAccessor> && ArithmeticSharedMemoryAccessor<ScratchAccessor>)
    static void __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor)
    {
        impl::reduce<Config,BinOp,Config::LevelCount,device_capabilities> fn;
        fn.template __call<DataAccessor,ScratchAccessor>(dataAccessor, scratchAccessor);
    }
};

template<class Config, class BinOp, class device_capabilities=void>
struct inclusive_scan
{
    template<class DataAccessor, class ScratchAccessor NBL_FUNC_REQUIRES(ArithmeticDataAccessor<DataAccessor> && ArithmeticSharedMemoryAccessor<ScratchAccessor>)
    static void __call(NBL_REF_ARG(DataAccessor) dataAccessor, NBL_REF_ARG(ScratchAccessor) scratchAccessor)
    {
        impl::scan<Config,BinOp,false,Config::LevelCount,device_capabilities> fn;
        fn.template __call<DataAccessor,ScratchAccessor>(dataAccessor, scratchAccessor);
    }
};

template<class Config, class BinOp, class device_capabilities=void>
struct exclusive_scan
{
    template<class DataAccessor, class ScratchAccessor NBL_FUNC_REQUIRES(ArithmeticDataAccessor<DataAccessor> && ArithmeticSharedMemoryAccessor<ScratchAccessor>)
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
