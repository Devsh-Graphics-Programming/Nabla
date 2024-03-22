// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_HLSL_EXT_FULL_SCREEN_TRIANGLE_S_VERTEX_ATTRIBUTE_H_
#define _NBL_HLSL_EXT_FULL_SCREEN_TRIANGLE_S_VERTEX_ATTRIBUTE_H_

namespace nbl
{
namespace hlsl
{
namespace ext
{
namespace FullScreenTriangle
{

struct SVertexAttributes
{
    [[vk::location(0)]] float32_t2 uv : TEXCOORD0;
};

}
}
}
}
#endif