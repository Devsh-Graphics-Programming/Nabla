// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core

layout(location = 0) in vec3 color; 
layout(location = 0) out vec4 pixelColor;

void main()
{
    pixelColor = vec4(color,1.0);
}