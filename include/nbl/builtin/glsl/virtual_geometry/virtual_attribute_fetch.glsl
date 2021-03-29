// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/builtin/glsl/format/decode.glsl"
#include "nbl/buildin/glsl/virtual_geometry/virtualAttribute.glsl"

vec3 nbl_glsl_VG_vertexFetch_RGB32_SFLOAT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_vertexFetch3f(attr,vtxID));
}

vec3 nbl_glsl_VG_vertexFetch_RG32_SFLOAT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_vertexFetch2f(attr,vtxID));
}

vec4 nbl_glsl_VG_vertexFetch_RGB10A2_SNORM(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_decodeRGB10A2_SNORM(nbl_glsl_VG_vertexFetch1u(attr,vtxID));
}