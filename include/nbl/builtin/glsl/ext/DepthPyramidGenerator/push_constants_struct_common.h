// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_GLSL_EXT_DEPTH_PYRAMID_GENERATOR_PUSH_CONSTANTS_STRUCT_COMMON_H_INCLUDED_
#define _NBL_GLSL_EXT_DEPTH_PYRAMID_GENERATOR_PUSH_CONSTANTS_STRUCT_COMMON_H_INCLUDED_

#ifdef __cplusplus
#define uint uint32_t

namespace nbl
{
namespace ext
{
namespace DepthPyramidGenerator
{

struct uvec2
{
    uvec2() = default;
    uvec2(uint32_t _x, uint32_t _y) : x(_x), y(_y) {}

    uint32_t x;
    uint32_t y;
};

}
}
}
#endif

struct nbl_glsl_depthPyramid_PushConstantsData
{
#ifdef __cplusplus
    nbl::ext::DepthPyramidGenerator::uvec2 mainDispatchFirstMipExtent;
    nbl::ext::DepthPyramidGenerator::uvec2 virtualDispatchFirstMipExtent;
#else
    uvec2 mainDispatchFirstMipExtent;
    uvec2 virtualDispatchFirstMipExtent;
#endif
    uint mainDispatchMipCnt;
    uint virtualDispatchMipCnt;
    uint maxMetaZLayerCnt;
    uint virtualDispatchIndex;
    uint sourceImageIsDepthOriginalDepthBuffer;

};

#ifdef __cplusplus
#undef uint
#endif

#endif