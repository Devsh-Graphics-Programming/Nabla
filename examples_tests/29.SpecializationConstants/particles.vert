// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core

layout (location = 0) in vec3 vPos;

#include <nbl/builtin/glsl/utils/common.glsl>
#include <nbl/builtin/glsl/utils/transform.glsl>

layout (push_constant) uniform UBO
{
    mat4 MVP;
} CamData;

void main()
{
	gl_Position = nbl_glsl_pseudoMul4x4with3x1(CamData.MVP, vPos);
}