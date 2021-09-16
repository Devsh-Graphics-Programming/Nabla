// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core

layout(location = 0)in vec3 cameraPos;
layout(location = 1)in vec3 cameraNormal;

layout(location = 0) out vec4 color;

void main()
{
    vec3 fragToEye = normalize(-cameraPos);
    color = vec4(vec3(clamp(dot(fragToEye, cameraNormal), 0.0, 1.0)), 1.0);
}
