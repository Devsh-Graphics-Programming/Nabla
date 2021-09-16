// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core
//#extension GL_GOOGLE_include_directive : require

layout(location = 0) in vec4 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 3) in vec3 vNormal;

layout(location = 0)out vec3 cameraPos;
layout(location = 1)out vec3 cameraNormal;

#include <nbl/builtin/glsl/utils/common.glsl>
#include <nbl/builtin/glsl/utils/transform.glsl>

layout (set = 0, binding = 0, row_major, std140) uniform UBO 
{
    nbl_glsl_SBasicViewParameters params;
} camData;

void main()
{
    gl_Position = camData.params.MVP*vPos;
    cameraPos = camData.params.MV*vPos;
    // no scaling, so it will work
    cameraNormal = normalize(camData.params.MV * vec4(vNormal, 0.0)); //have to normalize twice because of normal quantization
}
