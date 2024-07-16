#version 430 core

// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// vertex shader is provided by the fullScreenTriangle extension

layout (set = 3, binding = 0) uniform usampler2DArray tex0;

layout(location = 0) in vec2 TexCoord;

layout(location = 0) out vec4 pixelColor;

void main()
{
    const uint packed = texture(tex0, vec3(TexCoord.x, TexCoord.y, 1.0)).r;
    pixelColor = unpackUnorm4x8(packed);
}