// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 330 core
layout(location = 0) in vec4 posAndTC;

out vec2 TexCoord;

void main()
{
    gl_Position = vec4(posAndTC.xy,0.0,1.0); //only thing preventing the shader from being core-compliant
    TexCoord = posAndTC.zw;
}

