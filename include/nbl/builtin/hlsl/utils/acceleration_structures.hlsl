
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_UTILS_ACCELERATION_STRUCTURES_INCLUDED_
#define _NBL_BUILTIN_HLSL_UTILS_ACCELERATION_STRUCTURES_INCLUDED_


namespace nbl
{
namespace hlsl
{


// Use for Indirect Builds
struct BuildRangeInfo
{
    uint    primitiveCount;
    uint    primitiveOffset;
    uint    firstVertex;
    uint    transformOffset;

    #ifdef __cplusplus
    auto operator<=>(const BuildRangeInfo&) const = default;
    #endif // __cplusplus
};


}
}

#endif