// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core

in vec3 Normal;
in vec3 lightDir;
in vec2 uv;

layout(location = 0) out vec4 pixelColor;

layout(binding = 0) uniform sampler2D tex;

void main()
{
    pixelColor = texture(tex, uv) * max(dot(Normal, normalize(lightDir)), 0.);
}
