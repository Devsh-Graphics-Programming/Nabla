// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core

layout(set = 3, binding = 0) uniform sampler2D albedo;

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 pixelColor;

void main()
{
    pixelColor = texture(albedo,uv);
}
		