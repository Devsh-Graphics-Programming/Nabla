// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_VIRTUAL_GEOMETRY_VIRTUAL_ATTRIBUTE_FETCH_INCLUDED_
#define _NBL_BUILTIN_GLSL_VIRTUAL_GEOMETRY_VIRTUAL_ATTRIBUTE_FETCH_INCLUDED_

#include <nbl/builtin/glsl/format/decode.glsl>
#include <nbl/builtin/glsl/virtual_geometry/virtual_attribute.glsl>

//TODO: R8, R8G8, R8G8B8, RGBA8, R16, RGB16
// float nbl_glsl_VG_attribFetch_R8_UNORM(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
// {
//   return nbl_glsl_VG_attribFetch1f(attr, vertexID);
// }
// vec4 nbl_glsl_VG_attribFetch_RGBA8_SFLOAT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
// {
// #ifdef _NBL_VG_USE_SSBO
//   return unpackUnorm4x8(nbl_glsl_VG_attribFetch1u(attr, vertexID))
// #else
// #endif
// }

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC4)) || _NBL_VG_FLOAT_BUFFERS_COUNT
vec4 nbl_glsl_VG_attribFetch_RGBA32_SFLOAT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch4f(attr, vertexID);
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC3)) || _NBL_VG_FLOAT_BUFFERS_COUNT
vec3 nbl_glsl_VG_attribFetch_RGB32_SFLOAT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch3f(attr, vertexID);
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC2)) || _NBL_VG_FLOAT_BUFFERS_COUNT
vec2 nbl_glsl_VG_attribFetch_RG32_SFLOAT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch2f(attr, vertexID);
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UINT)) || _NBL_VG_FLOAT_BUFFERS_COUNT
float nbl_glsl_VG_attribFetch_R32_SFLOAT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch1f(attr, vertexID);
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC2)) || _NBL_VG_FLOAT_BUFFERS_COUNT
vec4 nbl_glsl_VG_attribFetch_RGBA16_SFLOAT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
#ifdef _NBL_VG_USE_SSBO
  uvec2 packed = nbl_glsl_VG_attribFetch2u(attr, vertexID);
  vec2 xy = unpackHalf2x16(packed[0]).xy;
  vec2 zw = unpackHalf2x16(packed[1]).xy;
  return vec4(xy, zw);
#else
  return nbl_glsl_VG_attribFetch4f(attr, vertexID);
#endif
}

vec4 nbl_glsl_VG_attribFetch_RGBA16_SNORM(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch_RGBA16_SFLOAT(attr, vertexID);
}

vec4 nbl_glsl_VG_attribFetch_RGBA16_UNORM(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch_RGBA16_SFLOAT(attr, vertexID);
}

vec4 nbl_glsl_VG_attribFetch_RGBA16_USCALED(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch_RGBA16_SFLOAT(attr, vertexID);
}

vec4 nbl_glsl_VG_attribFetch_RGBA16_SSCALED(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch_RGBA16_SFLOAT(attr, vertexID);
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UINT)) || _NBL_VG_FLOAT_BUFFERS_COUNT
vec2 nbl_glsl_VG_attribFetch_RG16_SFLOAT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
#ifdef _NBL_VG_USE_SSBO
  return unpackHalf2x16(nbl_glsl_VG_attribFetch1u(attr,vertexID));
#else
  return nbl_glsl_VG_attribFetch2f(attr, vertexID);
#endif
}

vec2 nbl_glsl_VG_attribFetch_RG16_SNORM(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch_RG16_SFLOAT(attr, vertexID);
}

vec2 nbl_glsl_VG_attribFetch_RG16_UNORM(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch_RG16_SFLOAT(attr, vertexID);
}

vec2 nbl_glsl_VG_attribFetch_RG16_USCALED(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch_RG16_SFLOAT(attr, vertexID);
}

vec2 nbl_glsl_VG_attribFetch_RG16_SSCALED(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch_RG16_SFLOAT(attr, vertexID);
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC4)) || _NBL_VG_INT_BUFFERS_COUNT
ivec4 nbl_glsl_VG_attribFetch_RGBA32_SINT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch4i(attr, vertexID);
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC3)) || _NBL_VG_INT_BUFFERS_COUNT
ivec3 nbl_glsl_VG_attribFetch_RGB32_SINT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch3i(attr, vertexID);
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC2)) || _NBL_VG_INT_BUFFERS_COUNT
ivec2 nbl_glsl_VG_attribFetch_RG32_SINT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch2i(attr, vertexID);
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UINT)) || _NBL_VG_INT_BUFFERS_COUNT
int nbl_glsl_VG_attribFetch_R32_SINT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch1i(attr, vertexID);
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC4)) || _NBL_VG_UINT_BUFFERS_COUNT
uvec4 nbl_glsl_VG_attribFetch_RGBA32_UINT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch4u(attr, vertexID);
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC3)) || _NBL_VG_UINT_BUFFERS_COUNT
uvec3 nbl_glsl_VG_attribFetch_RGB32_UINT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch3u(attr, vertexID);
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC2)) || _NBL_VG_UINT_BUFFERS_COUNT
uvec2 nbl_glsl_VG_attribFetch_RG32_UINT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch2u(attr, vertexID);
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UINT)) || _NBL_VG_UINT_BUFFERS_COUNT
uint nbl_glsl_VG_attribFetch_R32_UINT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch1u(attr, vertexID);
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UINT)) || _NBL_VG_UINT_BUFFERS_COUNT
vec4 nbl_glsl_VG_attribFetch_RGB10A2_SNORM(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_decodeRGB10A2_SNORM(nbl_glsl_VG_attribFetch1u(attr, vertexID));
}
#endif

#endif