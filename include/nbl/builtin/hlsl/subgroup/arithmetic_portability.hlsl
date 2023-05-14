// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_INCLUDED_

#ifndef NBL_GL_KHR_shader_subgroup_arithmetic
#include <nbl/builtin/hlsl/subgroup/basic_portability.hlsl>
#endif

namespace nbl
{
namespace hlsl
{
namespace subgroup
{

#ifdef NBL_GL_KHR_shader_subgroup_arithmetic
namespace native
{

template<typename T, class Binop>
struct reduction;
template<typename T, class Binop>
struct exclusive_scan;
template<typename T, class Binop>
struct inclusive_scan;

}
#endif

namespace portability
{

// PORTABILITY BINOP DECLARATIONS
template<typename T, class Binop, class ScratchAccessor>
struct reduction;
template<typename T, class Binop, class ScratchAccessor>
struct inclusive_scan;
template<typename T, class Binop, class ScratchAccessor>
struct exclusive_scan;

}

template<typename T, class Binop, class ScratchAccessor>
struct reduction
{
    T operator()(const T x)
    { // REVIEW: Should these extension headers have the GL name?
    #ifdef NBL_GL_KHR_shader_subgroup_arithmetic
        native::reduction<T, Binop> reduce;
        return reduce(x);
    #else
        return portability::reduction<Binop, ScratchAccessor>::create()(x);
    #endif
    }
};

template<typename T, class Binop, class ScratchAccessor>
struct exclusive_scan
{
    T operator()(const T x)
    {
    #ifdef NBL_GL_KHR_shader_subgroup_arithmetic
        native::exclusive_scan<T, Binop> scan;
        return scan(x);
    #else
        portability::exclusive_scan<Binop, ScratchAccessor>::create()(x);
    #endif
    }
};

template<typename T, class Binop, class ScratchAccessor>
struct inclusive_scan
{
    T operator()(const T x)
    {
    #ifdef NBL_GL_KHR_shader_subgroup_arithmetic
        native::inclusive_scan<T, Binop> scan;
        return scan(x);
    #else
        portability::inclusive_scan<Binop, ScratchAccessor>::create()(x);
    #endif
    }
};

}
}
}

#include <nbl/builtin/hlsl/subgroup/arithmetic_portability_impl.hlsl>

#endif