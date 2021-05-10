// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_BARYCENTRIC_VERT_INCLUDED_
#define _NBL_BUILTIN_GLSL_BARYCENTRIC_VERT_INCLUDED_

void nbl_glsl_barycentric_vert_set(in vec3 pos);

// TODO: Check for Nvidia Pascal Barycentric extension
// TODO: Check for AMD Barycentric extension
#if 0||0


void nbl_glsl_barycentric_vert_set(in vec3 pos) {} // noop


#else


#ifndef NBL_GLSL_BARYCENTRIC_VERT_POS_OUTPUT
#define NBL_GLSL_BARYCENTRIC_VERT_POS_OUTPUT
#ifndef NBL_GLSL_BARYCENTRIC_VERT_POS_OUTPUT_LOC
#define NBL_GLSL_BARYCENTRIC_VERT_POS_OUTPUT_LOC 0
#endif
layout(location = NBL_GLSL_BARYCENTRIC_VERT_POS_OUTPUT_LOC) out vec3 nbl_glsl_barycentric_vert_pos;
#endif
#ifndef NBL_GLSL_BARYCENTRIC_VERT_PROVOKINGPOS_OUTPUT
#define NBL_GLSL_BARYCENTRIC_VERT_PROVOKINGPOS_OUTPUT
#ifndef NBL_GLSL_BARYCENTRIC_VERT_PROVOKINGPOS_OUTPUT_LOC
#define NBL_GLSL_BARYCENTRIC_VERT_PROVOKINGPOS_OUTPUT_LOC 1
#endif
layout(location = NBL_GLSL_BARYCENTRIC_VERT_PROVOKINGPOS_OUTPUT_LOC) flat out vec3 nbl_glsl_barycentric_vert_provokingPos;
#endif

void nbl_glsl_barycentric_vert_set(in vec3 pos)
{
    nbl_glsl_barycentric_vert_pos = pos;
    nbl_glsl_barycentric_vert_provokingPos = pos;
}

#endif


#endif