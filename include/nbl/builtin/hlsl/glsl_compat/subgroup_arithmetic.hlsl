// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_GLSL_COMPAT_SUBGROUP_ARITHMETIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_GLSL_COMPAT_SUBGROUP_ARITHMETIC_INCLUDED_

#include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"

namespace nbl
{
namespace hlsl
{
namespace glsl
{
    
// TODO: @Hazardu this will need touching up in the impl when the `spirv::groupXXX` will change names to a predictable pattern from SPIR-V
// TODO: Furthermore you'll need `bitfieldExtract`-like struct dispatcher to choose between int/float add/mul and sint/uint/float min/max
template<typename T>
T subgroupAdd(T value) {
    return spirv::groupNonUniformIAdd_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationReduce, value);
}
template<typename T>
T subgroupInclusiveAdd(T value) {
    return spirv::groupNonUniformIAdd_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationInclusiveScan, value);
}
template<typename T>
T subgroupExclusiveAdd(T value) {
    return spirv::groupNonUniformIAdd_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationExclusiveScan, value);
}

template<typename T>
T subgroupMul(T value) {
    return spirv::groupNonUniformIMul_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationReduce, value);
}
template<typename T>
T subgroupInclusiveMul(T value) {
    return spirv::groupNonUniformIMul_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationInclusiveScan, value);
}
template<typename T>
T subgroupExclusiveMul(T value) {
    return spirv::groupNonUniformIMul_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationExclusiveScan, value);
}

template<typename T>
T subgroupAnd(T value) {
    return spirv::groupNonUniformBitwiseAnd_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationReduce, value);
}
template<typename T>
T subgroupInclusiveAnd(T value) {
    return spirv::groupNonUniformBitwiseAnd_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationInclusiveScan, value);
}
template<typename T>
T subgroupExclusiveAnd(T value) {
    return spirv::groupNonUniformBitwiseAnd_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationExclusiveScan, value);
}

template<typename T>
T subgroupOr(T value) {
    return spirv::groupNonUniformBitwiseOr_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationReduce, value);
}
template<typename T>
T subgroupInclusiveOr(T value) {
    return spirv::groupNonUniformBitwiseOr_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationInclusiveScan, value);
}
template<typename T>
T subgroupExclusiveOr(T value) {
    return spirv::groupNonUniformBitwiseOr_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationExclusiveScan, value);
}

template<typename T>
T subgroupXor(T value) {
    return spirv::groupNonUniformBitwiseXor_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationReduce, value);
}
template<typename T>
T subgroupInclusiveXor(T value) {
    return spirv::groupNonUniformBitwiseXor_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationInclusiveScan, value);
}
template<typename T>
T subgroupExclusiveXor(T value) {
    return spirv::groupNonUniformBitwiseXor_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationExclusiveScan, value);
}

namespace impl
{

template<typename T, bool isSigned>
struct subgroupMin {};

template<typename T>
struct subgroupMin<T, true>
{
    static T __call(T val)
    {
        return spirv::groupNonUniformSMin_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationReduce, val);
    }
};

template<typename T>
struct subgroupMin<T, false>
{
    static T __call(T val)
    {
        return spirv::groupNonUniformUMin_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationReduce, val);
    } 
};

template<typename T, bool isSigned>
struct subgroupInclusiveMin {};

template<typename T>
struct subgroupInclusiveMin<T, true>
{
    static T __call(T val)
    {
        return spirv::groupNonUniformSMin_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationInclusiveScan, val);
    }
};

template<typename T>
struct subgroupInclusiveMin<T, false>
{
    static T __call(T val)
    {
        return spirv::groupNonUniformUMin_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationInclusiveScan, val);
    } 
};

template<typename T, bool isSigned>
struct subgroupExclusiveMin {};

template<typename T>
struct subgroupExclusiveMin<T, true>
{
    static T __call(T val)
    {
        return spirv::groupNonUniformSMin_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationExclusiveScan, val);
    }
};

template<typename T>
struct subgroupExclusiveMin<T, false>
{
    static T __call(T val)
    {
        return spirv::groupNonUniformUMin_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationExclusiveScan, val);
    } 
};

template<typename T, bool isSigned>
struct subgroupMax {};

template<typename T>
struct subgroupMax<T, true>
{
    static T __call(T val)
    {
        return spirv::groupNonUniformSMax_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationReduce, val);
    }
};

template<typename T>
struct subgroupMax<T, false>
{
    static T __call(T val)
    {
        return spirv::groupNonUniformUMax_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationReduce, val);
    } 
};

template<typename T, bool isSigned>
struct subgroupInclusiveMax {};

template<typename T>
struct subgroupInclusiveMax<T, true>
{
    static T __call(T val)
    {
        return spirv::groupNonUniformSMax_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationInclusiveScan, val);
    }
};

template<typename T>
struct subgroupInclusiveMax<T, false>
{
    static T __call(T val)
    {
        return spirv::groupNonUniformUMax_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationInclusiveScan, val);
    } 
};

template<typename T, bool isSigned>
struct subgroupExclusiveMax {};

template<typename T>
struct subgroupExclusiveMax<T, true>
{
    static T __call(T val)
    {
        return spirv::groupNonUniformSMax_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationExclusiveScan, val);
    }
};

template<typename T>
struct subgroupExclusiveMax<T, false>
{
    static T __call(T val)
    {
        return spirv::groupNonUniformUMax_GroupNonUniformArithmetic<T>(spv::ScopeSubgroup, spv::GroupOperationExclusiveScan, val);
    } 
};

}

template<typename T>
T subgroupMin(T val) {
    return impl::subgroupMin<T, is_signed<T>::value>::__call(val);
}
template<typename T>
T subgroupInclusiveMin(T val) {
    return impl::subgroupInclusiveMin<T, is_signed<T>::value>::__call(val);
}
template<typename T>
T subgroupExclusiveMin(T val) {
    return impl::subgroupExclusiveMin<T, is_signed<T>::value>::__call(val);
}

template<typename T>
T subgroupMax(T val) {
    return impl::subgroupMax<T, is_signed<T>::value>::__call(val);
}
template<typename T>
T subgroupInclusiveMax(T val) {
    return impl::subgroupInclusiveMax<T, is_signed<T>::value>::__call(val);
}
template<typename T>
T subgroupExclusiveMax(T val) {
    return impl::subgroupExclusiveMax<T, is_signed<T>::value>::__call(val);
}

}
}
}

#endif