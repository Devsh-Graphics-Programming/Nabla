// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_INCLUDED_


#include "nbl/builtin/hlsl/subgroup/basic.hlsl"

#include "nbl/builtin/hlsl/subgroup/arithmetic_portability_impl.hlsl"

namespace nbl
{
namespace hlsl
{
namespace subgroup
{

// TODO: when DXC SPIR-V output gets template aliases, we can get of og this forwarding
template<typename T, class Binop>
struct reduction
{
    T operator()(NBL_CONST_REF_ARG(T) x)
    {
    #ifdef NBL_GL_KHR_shader_subgroup_arithmetic
        native::reduction<T, Binop> reduce;
        return reduce(x);
    #else
        return portability::reduction<T, Binop>(x);
    #endif
    }
};

template<typename T, class Binop>
struct exclusive_scan
{
    T operator()(NBL_CONST_REF_ARG(T) x)
    {
    #ifdef NBL_GL_KHR_shader_subgroup_arithmetic
        native::exclusive_scan<T, Binop> scan;
        return scan(x);
    #else
        return portability::exclusive_scan<T, Binop>(x);
    #endif
    }
};

template<typename T, class Binop>
struct inclusive_scan
{
    T operator()(NBL_CONST_REF_ARG(T) x)
    {
    #ifdef NBL_GL_KHR_shader_subgroup_arithmetic
        native::inclusive_scan<T, Binop> scan;
        return scan(x);
    #else
        return portability::inclusive_scan<T, Binop>(x);
    #endif
    }
};

}
}
}

#endif