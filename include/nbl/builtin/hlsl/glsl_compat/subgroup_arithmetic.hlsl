// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_GLSL_COMPAT_SUBGROUP_ARITHMETIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_GLSL_COMPAT_SUBGROUP_ARITHMETIC_INCLUDED_

#include "nbl/builtin/hlsl/spirv_intrinsics/subgroup_arithmetic.hlsl"

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
    return spirv::groupAdd(spv::ScopeSubgroup, spv::GroupOperationReduce, value);
}
template<typename T>
T subgroupInclusiveAdd(T value) {
    return spirv::groupAdd(spv::ScopeSubgroup, spv::GroupOperationInclusiveScan, value);
}
template<typename T>
T subgroupExclusiveAdd(T value) {
    return spirv::groupAdd(spv::ScopeSubgroup, spv::GroupOperationExclusiveScan, value);
}

template<typename T>
T subgroupMul(T value) {
    return spirv::groupMul(spv::ScopeSubgroup, spv::GroupOperationReduce, value);
}
template<typename T>
T subgroupInclusiveMul(T value) {
    return spirv::groupMul(spv::ScopeSubgroup, spv::GroupOperationInclusiveScan, value);
}
template<typename T>
T subgroupExclusiveMul(T value) {
    return spirv::groupMul(spv::ScopeSubgroup, spv::GroupOperationExclusiveScan, value);
}

template<typename T>
T subgroupAnd(T value) {
    return spirv::groupBitwiseAnd(spv::ScopeSubgroup, spv::GroupOperationReduce, value);
}
template<typename T>
T subgroupInclusiveAnd(T value) {
    return spirv::groupBitwiseAnd(spv::ScopeSubgroup, spv::GroupOperationInclusiveScan, value);
}
template<typename T>
T subgroupExclusiveAnd(T value) {
    return spirv::groupBitwiseAnd(spv::ScopeSubgroup, spv::GroupOperationExclusiveScan, value);
}

template<typename T>
T subgroupOr(T value) {
    return spirv::groupBitwiseOr(spv::ScopeSubgroup, spv::GroupOperationReduce, value);
}
template<typename T>
T subgroupInclusiveOr(T value) {
    return spirv::groupBitwiseOr(spv::ScopeSubgroup, spv::GroupOperationInclusiveScan, value);
}
template<typename T>
T subgroupExclusiveOr(T value) {
    return spirv::groupBitwiseOr(spv::ScopeSubgroup, spv::GroupOperationExclusiveScan, value);
}

template<typename T>
T subgroupXor(T value) {
    return spirv::groupBitwiseXor(spv::ScopeSubgroup, spv::GroupOperationReduce, value);
}
template<typename T>
T subgroupInclusiveXor(T value) {
    return spirv::groupBitwiseXor(spv::ScopeSubgroup, spv::GroupOperationInclusiveScan, value);
}
template<typename T>
T subgroupExclusiveXor(T value) {
    return spirv::groupBitwiseXor(spv::ScopeSubgroup, spv::GroupOperationExclusiveScan, value);
}

template<typename T>
T subgroupMin(T value) {
    return spirv::groupBitwiseMin(spv::ScopeSubgroup, spv::GroupOperationReduce, value);
}
template<typename T>
T subgroupInclusiveMin(T value) {
    return spirv::groupBitwiseMin(spv::ScopeSubgroup, spv::GroupOperationInclusiveScan, value);
}
template<typename T>
T subgroupExclusiveMin(T value) {
    return spirv::groupBitwiseMin(spv::ScopeSubgroup, spv::GroupOperationExclusiveScan, value);
}

template<typename T>
T subgroupMax(T value) {
    return spirv::groupBitwiseMax(spv::ScopeSubgroup, spv::GroupOperationReduce, value);
}
template<typename T>
T subgroupInclusiveMax(T value) {
    return spirv::groupBitwiseMax(spv::ScopeSubgroup, spv::GroupOperationInclusiveScan, value);
}
template<typename T>
T subgroupExclusiveMax(T value) {
    return spirv::groupBitwiseMax(spv::ScopeSubgroup, spv::GroupOperationExclusiveScan, value);
}

}
}
}

#endif