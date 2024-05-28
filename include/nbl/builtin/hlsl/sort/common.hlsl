// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SORT_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_SORT_COMMON_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace sort
{

template<class Key>
struct CountingParameters
{
    static_assert(is_integral<Key>::value, "CountingParameters needs to be templated on integral type");

    uint32_t dataElementCount;
    uint32_t elementsPerWT;
    uint32_t workGroupIndex;
    Key minimum;
    Key maximum;
};


}
}
}
#endif