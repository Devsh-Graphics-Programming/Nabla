// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_TYPE_TRAITS_INCLUDED_
#define _NBL_BUILTIN_HLSL_TYPE_TRAITS_INCLUDED_

// REVIEW-519: We said 1:1 with STL, does that mean without nbl::hlsl namespace so that code can be interoperable with CPP?
//  Additionally there is a type_traits.hlsl header at the root folder. Delete?
namespace nbl
{
namespace hlsl
{

template<class T, T v>
struct integral_constant
{
    static const T value = v;
    using value_type = T;
    using type = integral_constant;
};

}
}
#endif