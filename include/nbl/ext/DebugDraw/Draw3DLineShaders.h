// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_DRAW_3D_LINE_SHADERS_INCLUDED_
#define _NBL_EXT_DRAW_3D_LINE_SHADERS_INCLUDED_

namespace nbl
{
namespace ext
{
namespace DebugDraw
{

static constexpr const auto Draw3DLineVertexShader = R"===(
#version 430 core

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec4 vCol;

layout( push_constant, row_major ) uniform PushConstants {
  mat4 vp;
} pc;
layout(location = 0) out vec4 Color;

void main()
{
    gl_Position = pc.vp * vec4(vPos, 1.0);
    Color = vCol;
}
)===";

static constexpr const auto Draw3DLineFragmentShader = R"===(
#version 430 core

layout(location = 0) in vec4 Color;

layout(location = 0) out vec4 pixelColor;

void main()
{
    pixelColor = Color;
}
)===";

} // namespace DebugDraw
} // namespace ext
} // namespace nbl

#endif
