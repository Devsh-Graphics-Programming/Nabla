// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 420 core
uniform mat4 MVP;

layout(location = 0) in vec3 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 2) in vec2 vTC;

out vec2 TexCoord;
out float height;


void main()
{
    gl_Position = MVP*vec4(vPos,1.0);
    TexCoord = vTC;
    height = vPos.y*0.5;
}
