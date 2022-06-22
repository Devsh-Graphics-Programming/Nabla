// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core

#include <nbl/builtin/glsl/utils/swapchain_transform.glsl>

const vec2 pos[3] = vec2[3](vec2(-1.0, 1.0),vec2(-1.0,-3.0),vec2( 3.0, 1.0));
const vec2 tc[3] = vec2[3](vec2( 0.0, 0.0),vec2( 0.0, 2.0),vec2( 2.0, 0.0));

layout(location = 0) out vec2 TexCoord;
 
layout (push_constant) uniform pushConstants
{
	layout (offset = 0) uint swapchainTransform;
} u_pushConstants;

void main()
{
    mat2 transform = nbl_glsl_swapchain_transform_postTransformMatrix(u_pushConstants.swapchainTransform);
    gl_Position = vec4(transform * pos[gl_VertexIndex], 0.0, 1.0);
    TexCoord = tc[gl_VertexIndex];
}