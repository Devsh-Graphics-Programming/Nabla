// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_ACCELERATION_STRUCTURES_INCLUDED_
#define _NBL_BUILTIN_HLSL_ACCELERATION_STRUCTURES_INCLUDED_


#include "nlb/builtin/hlsl/acceleration_structures.glsl"


namespace nbl
{
namespace hlsl
{
namespace acceleration_structures
{

// Use for Indirect Builds
struct BuildRangeInfo
{
    uint32_t primitiveCount;
    uint32_t primitiveOffset;
    uint32_t firstVertex;
    uint32_t transformOffset;

    #ifdef __cplusplus
    auto operator<=>(const BuildRangeInfo&) const = default;
    #endif // __cplusplus
};

}
}
}

#endif