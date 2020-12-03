// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 420 core
layout(binding = 0) uniform sampler2D tex0;

in vec3 Normal;
in vec2 TexCoord;
in vec3 lightDir;

layout(location = 0) out vec4 pixelColor;

void main()
{
    pixelColor = texture(tex0,TexCoord)*max(dot(normalize(Normal),normalize(lightDir)),0.0);
}
