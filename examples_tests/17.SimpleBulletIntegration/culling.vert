// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core

layout(location = 0) in vec4 vWorldMatPart0;
layout(location = 1) in vec4 vWorldMatPart1;
layout(location = 2) in vec4 vWorldMatPart2;

layout(location = 5) in uvec2 vNormalMatPart2;

out vec4 gWorldMatPart0;
out vec4 gWorldMatPart1;
out vec4 gWorldMatPart2;
out uint gInstanceColor;

void main()
{
    gWorldMatPart0 = vWorldMatPart0;
    gWorldMatPart1 = vWorldMatPart1;
    gWorldMatPart2 = vWorldMatPart2;
    gInstanceColor = vNormalMatPart2.y;
}
