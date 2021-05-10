// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 460 core

layout(push_constant, row_major) uniform PushConstants
{
	mat4 vp;
} pc;

layout(location = 0) in vec4 vPos;
layout(location = 2) in vec2 vTexCoord;
layout(location = 3) in vec3 vNormal;

layout(location = 0) out vec2 texCoord;
layout(location = 1) flat out uint drawID;

void main()
{
    gl_Position = pc.vp * vPos;
    
    drawID = gl_DrawID;
    texCoord = vTexCoord;
}
