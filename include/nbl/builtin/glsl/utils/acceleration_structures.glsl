// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_GLSL_UTILS_ACCELERATION_STRUCTURES_INCLUDED_
#define _NBL_BUILTIN_GLSL_UTILS_ACCELERATION_STRUCTURES_INCLUDED_

// Use for Indirect Builds
struct nbl_glsl_BuildRangeInfo {
    uint    primitiveCount;
    uint    primitiveOffset;
    uint    firstVertex;
    uint    transformOffset;

    #ifdef __cplusplus
    auto operator<=>(const nbl_glsl_BuildRangeInfo&) const = default;
    #endif // __cplusplus
};

#endif