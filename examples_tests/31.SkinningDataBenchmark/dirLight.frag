// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core
layout(location = 0) in vec3 vNormal;

layout(location = 0) out vec4 pixelColor;

void main()
{
    vec3 normColor = vec3(vNormal.x) * 0.5 + vec3(0.5);
    pixelColor = vec4(normColor, 1.0);
}
