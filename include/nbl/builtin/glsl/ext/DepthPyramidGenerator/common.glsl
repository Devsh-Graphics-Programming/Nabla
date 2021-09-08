// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_GLSL_EXT_DEPTH_PYRAMID_GENERATOR_COMMON_INCLUDED_
#define _NBL_GLSL_EXT_DEPTH_PYRAMID_GENERATOR_COMMON_INCLUDED_

//TODO: this is used in cpp code too, make it common for both cpp and glsl
struct PushConstantsData
{
    uint mainDispatchMipCnt;
    uint virtualDispatchMipCnt;
    uint maxMetaZLayerCnt;
    uint virtualDispatchIndex;
    uvec2 mainDispatchFirstMipExtent;
    uvec2 virtualDispatchFirstMipExtent;
};

layout(push_constant) uniform PushConstants
{
    PushConstantsData data;
} pc;

#endif