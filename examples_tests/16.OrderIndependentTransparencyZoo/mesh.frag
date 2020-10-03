// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 420 core
layout(early_fragment_tests) in;

layout(binding = 0) uniform sampler2D tex0;
uniform vec3 selfPos;

in vec2 TexCoord;
in float height;

layout(location = 0) out vec4 pixelColor;

void main()
{
    float alpha = min(height*(length(selfPos.xz)*0.02+1.0)*0.03,1.0);
    pixelColor = vec4(texture(tex0,TexCoord).rgb,alpha);
}
