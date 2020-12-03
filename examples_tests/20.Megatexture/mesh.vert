// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core
uniform mat4 MVP;

layout(location = 0) in vec4 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 2) in vec2 vTexCoord;

out vec2 TexCoord;

void main()
{
    gl_Position = MVP*vPos;
	TexCoord = vTexCoord;
}
