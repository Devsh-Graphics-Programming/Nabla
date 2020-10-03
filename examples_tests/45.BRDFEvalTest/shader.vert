// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core

layout (location = 0) in vec3 Pos;
layout (location = 3) in vec3 Normal;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outPos;
layout (location = 2) flat out float outAlpha;

layout (push_constant, row_major) uniform PC {
    mat4 VP;
} pc;

vec3 to_right_hand(in vec3 v)
{
    return v*vec3(-1.0,1.0,1.0);
}

void main()
{
    vec3 pos = to_right_hand(Pos) + vec3(float(gl_InstanceIndex)*1.5, 0.0, -1.0);
    outPos = pos;
    gl_Position = pc.VP*vec4(pos, 1.0);
    outNormal = to_right_hand(normalize(Normal));
    outAlpha = float(gl_InstanceIndex+1)*0.1;
}