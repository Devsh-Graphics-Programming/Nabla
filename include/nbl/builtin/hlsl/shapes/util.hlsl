// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SHAPES_UTIL_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_UTIL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{
namespace shapes
{
namespace util
{

namespace impl
{
template<typename T>
struct intersect_helper;
template<typename T>
struct union_helper;
}

template<typename T>
T intersect(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs) {return intersect_helper<T>::__call(lhs,rhs);} 
template<typename T>
T union(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs) {return union_helper<T>::__call(lhs,rhs);}

}
}
}
}

#endif