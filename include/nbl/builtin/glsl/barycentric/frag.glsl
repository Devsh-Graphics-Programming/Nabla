// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_BARYCENTRIC_FRAG_INCLUDED_
#define _NBL_BUILTIN_GLSL_BARYCENTRIC_FRAG_INCLUDED_


vec2 nbl_glsl_barycentric_frag_get();

// TODO: Check for Nvidia Pascal Barycentric extension
#if 0


//


// TODO: Check for AMD Barycentric extension
#elif 0


//


#else


#ifndef NBL_GLSL_BARYCENTRIC_FRAG_POS_INPUT
#define NBL_GLSL_BARYCENTRIC_FRAG_POS_INPUT
#ifndef NBL_GLSL_BARYCENTRIC_FRAG_POS_INPUT_LOC
#define NBL_GLSL_BARYCENTRIC_FRAG_POS_INPUT_LOC 0
#endif
layout(location = NBL_GLSL_BARYCENTRIC_FRAG_POS_INPUT_LOC) in vec3 nbl_glsl_barycentric_frag_pos;
#endif
#ifndef NBL_GLSL_BARYCENTRIC_FRAG_PROVOKINGPOS_INPUT
#define NBL_GLSL_BARYCENTRIC_FRAG_PROVOKINGPOS_INPUT
#ifndef NBL_GLSL_BARYCENTRIC_FRAG_PROVOKINGPOS_INPUT_LOC
#define NBL_GLSL_BARYCENTRIC_FRAG_PROVOKINGPOS_INPUT_LOC 1
#endif
layout(location = NBL_GLSL_BARYCENTRIC_FRAG_PROVOKINGPOS_INPUT_LOC) flat in vec3 nbl_glsl_barycentric_frag_provokingPos;
#endif

// these will need to be defined by the user
uint nbl_glsl_barycentric_frag_getDrawID();
vec3 nbl_glsl_barycentric_frag_getVertexPos(in uint drawID, in uint primID, in uint primsVx);

#ifndef _NBL_BUILTIN_GLSL_BARYCENTRIC_FRAG_GET_DEFINED_
#include <nbl/builtin/glsl/barycentric/utils.glsl>
vec2 nbl_glsl_barycentric_frag_get()
{
    return nbl_glsl_barycentric_reconstructBarycentrics(nbl_glsl_barycentric_frag_pos,mat3(
        nbl_glsl_barycentric_frag_provokingPos,
        nbl_glsl_barycentric_frag_getVertexPos(nbl_glsl_barycentric_frag_getDrawID(),gl_PrimitiveID,1u),
        nbl_glsl_barycentric_frag_getVertexPos(nbl_glsl_barycentric_frag_getDrawID(),gl_PrimitiveID,2u)
    ));
}
#define _NBL_BUILTIN_GLSL_BARYCENTRIC_FRAG_GET_DEFINED_
#endif


#endif


#endif