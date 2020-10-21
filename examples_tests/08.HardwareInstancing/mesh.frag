// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 330 core

in vec4 Color; //per vertex output color, will be interpolated across the triangle
in vec3 Normal;
in vec3 lightDir;

layout(location = 0) out vec4 pixelColor;

void main()
{
    pixelColor = Color*max(dot(normalize(Normal),normalize(lightDir)),0.0);
}
