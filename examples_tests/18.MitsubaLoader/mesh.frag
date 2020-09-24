// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core
layout(location = 1) uniform vec3 color;
layout(location = 2) uniform uint nasty;

layout(binding = 0) uniform sampler2D reflectance;

in vec2 uv;
in vec3 Normal;

layout(location = 0) out vec4 pixelColor;

#define MITS_TWO_SIDED		0x80000000u
#define MITS_USE_TEXTURE	0x40000000u

void main()
{
	if ((nasty&MITS_USE_TEXTURE) == MITS_USE_TEXTURE)
	    pixelColor = texture(reflectance,uv);
	else
		pixelColor = vec4(color,1.0);
}
