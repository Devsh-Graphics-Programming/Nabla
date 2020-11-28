// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core
#extension GL_GOOGLE_include_directive : require

layout(location = 0) in vec3 vPos; 
layout(location = 3) in vec3 vNormal;

#include <irr/builtin/glsl/utils/common.glsl>
#include <irr/builtin/glsl/utils/transform.glsl>

layout (set = 1, binding = 0, row_major, std140) uniform UBO 
{
    nbl_glsl_SBasicViewParameters params;
} CamData;

layout(location = 0) out vec3 color;

void main()
{
    gl_Position = nbl_glsl_pseudoMul4x4with3x1(CamData.params.MVP, vPos);
    color = vNormal*0.5+vec3(0.5);
}