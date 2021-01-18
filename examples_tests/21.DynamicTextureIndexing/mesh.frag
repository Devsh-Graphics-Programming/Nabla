// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 460 core

struct SSBOContents
{
	uint irrevelant[5];
	uint diffBind;
	uint bumpBind;
};

layout(set = 0, binding = 0, std430) restrict readonly buffer SSBO
{
	SSBOContents ssboContents[];
};

layout(set = 0, binding = 1) uniform sampler2D tex[16];

layout(location = 0) in vec2 texCoord;
layout(location = 1) flat in uint drawID;

layout(location = 0) out vec4 pixelColor;

void main()
{
	pixelColor = texture(tex[ssboContents[drawID].diffBind], texCoord);
}
