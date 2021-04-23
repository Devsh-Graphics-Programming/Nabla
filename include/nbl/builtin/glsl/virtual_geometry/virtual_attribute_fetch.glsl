// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_VIRTUAL_GEOMETRY_VIRTUAL_ATTRIBUTE_FETCH_INCLUDED_
#define _NBL_BUILTIN_GLSL_VIRTUAL_GEOMETRY_VIRTUAL_ATTRIBUTE_FETCH_INCLUDED_

#include <nbl/builtin/glsl/format/decode.glsl>
#include <nbl/builtin/glsl/virtual_geometry/virtual_attribute.glsl>

vec3 nbl_glsl_VG_attribFetch_RGB32_SFLOAT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch3f(attr, vertexID);
}

vec2 nbl_glsl_VG_attribFetch_RG32_SFLOAT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch2f(attr, vertexID);
}

vec4 nbl_glsl_VG_attribFetch_RGB10A2_SNORM(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_decodeRGB10A2_SNORM(nbl_glsl_VG_attribFetch1u(attr, vertexID));
}

#endif