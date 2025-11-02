// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_PAIR_INCLUDED_
#define _NBL_BUILTIN_HLSL_PAIR_INCLUDED_

#include "nbl/builtin/hlsl/type_traits.hlsl"

namespace nbl
{
namespace hlsl
{

template<typename T1, typename T2>
struct pair
{
    using first_type = T1;
    using second_type = T2;

    first_type first;
    second_type second;
};


// Helper to make a pair (similar to std::make_pair)
template<typename T1, typename T2>
pair<T1, T2> make_pair(T1 f, T2 s)
{
    pair<T1, T2> p;
    p.first = f;
    p.second = s;
    return p;
}

}
}

#endif
