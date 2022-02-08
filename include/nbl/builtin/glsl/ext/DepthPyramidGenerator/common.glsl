// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_GLSL_EXT_DEPTH_PYRAMID_GENERATOR_COMMON_INCLUDED_
#define _NBL_GLSL_EXT_DEPTH_PYRAMID_GENERATOR_COMMON_INCLUDED_

#include <nbl/builtin/glsl/ext/DepthPyramidGenerator/push_constants_struct_common.h>

layout(push_constant) uniform PushConstants
{
    nbl_glsl_depthPyramid_PushConstantsData data;
} pc;

#endif