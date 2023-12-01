// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_ACCELERATION_STRUCTURES_INCLUDED_
#define _NBL_BUILTIN_HLSL_ACCELERATION_STRUCTURES_INCLUDED_


#include "nbl/builtin/hlsl/cpp_compat.hlsl"


namespace nbl
{
namespace hlsl
{
namespace acceleration_structures
{

// you can actually use the same struct for both top level and bottom level indirect builds since they have similar fields in the first few bytes, just remember to set the correct strides

namespace bottom_level
{
// TODO: when Vulkan supports unions, do separate structs for AABB and Triangles and union them
struct BuildRangeInfo
{
    uint32_t primitiveCount; // needs to stay constant across updates
    uint32_t primitiveByteOffset;
    // following are only relevant for Triangle Geometries in BLASes
    uint32_t firstVertex; // needs to stay constant across updates
    uint32_t transformByteOffset;

    #ifdef __cplusplus
    auto operator<=>(const BuildRangeInfo&) const = default;
    #endif // __cplusplus
};
}

namespace top_level
{
// Vulkan actually wants to use the same struct as an AABB BLAS, and reinterpret the `primitive` as `instance
struct BuildRangeInfo
{
    uint32_t instanceCount; // needs to stay constant across updates
    uint32_t instanceByteOffset;

    #ifdef __cplusplus
    auto operator<=>(const BuildRangeInfo&) const = default;
    #endif // __cplusplus
};
}

}
}
}

#endif