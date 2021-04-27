// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_VIRTUAL_GEOMETRY_VIRTUAL_ATTRIBUTE_FETCH_INCLUDED_
#define _NBL_BUILTIN_GLSL_VIRTUAL_GEOMETRY_VIRTUAL_ATTRIBUTE_FETCH_INCLUDED_

#include <nbl/builtin/glsl/format/decode.glsl>
#include <nbl/builtin/glsl/virtual_geometry/virtual_attribute.glsl>

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
  vec2 xy = unpackHalf2x16(packed.x);
  vec2 zw = unpackHalf2x16(packed.y);
  return vec4(xy, zw);
#else
  return nbl_glsl_VG_attribFetch4f(attr, vertexID);
#endif
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC2)) || _NBL_VG_FLOAT_BUFFERS_COUNT
vec4 nbl_glsl_VG_attribFetch_RGBA16_SNORM(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
#ifdef _NBL_VG_USE_SSBO
  uvec2 packed = nbl_glsl_VG_attribFetch2u(attr, vertexID);
  vec2 xy = unpackSnorm2x16(packed.x);
  vec2 zw = unpackSnorm2x16(packed.y);
  return vec4(xy, zw);
#else
  return nbl_glsl_VG_attribFetch4f(attr, vertexID);
#endif
}

vec4 nbl_glsl_VG_attribFetch_RGBA16_UNORM(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
#ifdef _NBL_VG_USE_SSBO
  uvec2 packed = nbl_glsl_VG_attribFetch2u(attr, vertexID);
  vec2 xy = unpackUnorm2x16(packed.x);
  vec2 zw = unpackUnorm2x16(packed.y);
  return vec4(xy, zw);
#else
  return nbl_glsl_VG_attribFetch4f(attr, vertexID);
#endif
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UINT)) || _NBL_VG_FLOAT_BUFFERS_COUNT
vec2 nbl_glsl_VG_attribFetch_RG16_SFLOAT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
#ifdef _NBL_VG_USE_SSBO
  return unpackHalf2x16(nbl_glsl_VG_attribFetch1u(attr, vertexID));
#else
  return nbl_glsl_VG_attribFetch2f(attr, vertexID);
#endif
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UINT)) || _NBL_VG_FLOAT_BUFFERS_COUNT
vec2 nbl_glsl_VG_attribFetch_RG16_SNORM(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
#ifdef _NBL_VG_USE_SSBO
  return unpackSnorm2x16(nbl_glsl_VG_attribFetch1u(attr, vertexID));
#else
  return nbl_glsl_VG_attribFetch2f(attr, vertexID);
#endif
}

vec2 nbl_glsl_VG_attribFetch_RG16_UNORM(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
#ifdef _NBL_VG_USE_SSBO
  return unpackUnorm2x16(nbl_glsl_VG_attribFetch1u(attr, vertexID));
#else
  return nbl_glsl_VG_attribFetch2f(attr, vertexID);
#endif
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UINT)) || _NBL_VG_FLOAT_BUFFERS_COUNT
vec4 nbl_glsl_VG_attribFetch_RGBA8_SNORM(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
#ifdef _NBL_VG_USE_SSBO
  return unpackSnorm4x8(nbl_glsl_VG_attribFetch1u(attr, vertexID));
#else
  return nbl_glsl_VG_attribFetch4f(attr, vertexID);
#endif
}

vec4 nbl_glsl_VG_attribFetch_RGBA8_UNORM(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
#ifdef _NBL_VG_USE_SSBO
  return unpackUnorm4x8(nbl_glsl_VG_attribFetch1u(attr, vertexID));
#else
  return nbl_glsl_VG_attribFetch4f(attr, vertexID);
#endif
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

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UINT)) || _NBL_VG_UINT_BUFFERS_COUNT
vec4 nbl_glsl_VG_attribFetch_RGB10A2_UNORM(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_decodeRGB10A2_UNORM(nbl_glsl_VG_attribFetch1u(attr, vertexID));
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UINT)) || _NBL_VG_SINT_BUFFERS_COUNT
ivec4 nbl_glsl_VG_attribFetch_RGBA8_SINT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
#ifdef _NBL_VG_USE_SSBO
  uint packed = nbl_glsl_VG_attribFetch1u(attr, vertexID);
  uvec4 tmp;
  tmp.x = bitfieldExtract(packed, 0, 8);
  tmp.y = bitfieldExtract(packed, 8, 8);
  tmp.z = bitfieldExtract(packed, 16, 8);
  tmp.w = bitfieldExtract(packed, 24, 8);
  return ivec4(tmp) - ivec4(128);
#else
  return nbl_glsl_VG_attribFetch4i(attr, vertexID);
#endif
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UINT)) || _NBL_VG_SINT_BUFFERS_COUNT
uvec4 nbl_glsl_VG_attribFetch_RGBA8_UINT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
#ifdef _NBL_VG_USE_SSBO
  uint packed = nbl_glsl_VG_attribFetch1u(attr, vertexID);
  uvec4 result;
  result.x = bitfieldExtract(packed, 0, 8);
  result.y = bitfieldExtract(packed, 8, 8);
  result.z = bitfieldExtract(packed, 16, 8);
  result.w = bitfieldExtract(packed, 24, 8);
  return result;
#else
  return nbl_glsl_VG_attribFetch4u(attr, vertexID);
#endif
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UINT)) || _NBL_VG_INT_BUFFERS_COUNT
ivec4 nbl_glsl_VG_attribFetch_RGBA16_SINT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
#ifdef _NBL_VG_USE_SSBO
  uvec2 packed = nbl_glsl_VG_attribFetch2u(attr, vertexID);
  uvec4 tmp;
  tmp.x = bitfieldExtract(packed.x, 0, 16);
  tmp.y = bitfieldExtract(packed.x, 16, 16);
  tmp.z = bitfieldExtract(packed.y, 0, 16);
  tmp.w = bitfieldExtract(packed.y, 16, 16);
  return ivec4(tmp) - ivec4(32768);
#else
  return nbl_glsl_VG_attribFetch4i(attr, vertexID);
#endif
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UINT)) || _NBL_VG_UINT_BUFFERS_COUNT
uvec4 nbl_glsl_VG_attribFetch_RGBA16_UINT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
#ifdef _NBL_VG_USE_SSBO
  uvec2 packed = nbl_glsl_VG_attribFetch2u(attr, vertexID);
  uvec4 result;
  result.x = bitfieldExtract(packed.x, 0, 16);
  result.y = bitfieldExtract(packed.x, 16, 16);
  result.z = bitfieldExtract(packed.y, 0, 16);
  result.w = bitfieldExtract(packed.y, 16, 16);
  return result;
#else
  return nbl_glsl_VG_attribFetch4u(attr, vertexID);
#endif
}
#endif

#if _NBL_VG_FLOAT_BUFFERS_COUNT
float nbl_glsl_VG_attribFetch_R8_SNORM(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch1f(attr, vertexID);
}

float nbl_glsl_VG_attribFetch_R8_UNORM(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch1f(attr, vertexID);
}

vec2 nbl_glsl_VG_attribFetch_RG8_SNORM(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch2f(attr, vertexID);
}

vec2 nbl_glsl_VG_attribFetch_RG8_UNORM(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch2f(attr, vertexID);
}

vec3 nbl_glsl_VG_attribFetch_RGB8_SNORM(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch3f(attr, vertexID);
}

vec3 nbl_glsl_VG_attribFetch_RGB8_UNORM(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch3f(attr, vertexID);
}

float nbl_glsl_VG_attribFetch_R16_SFLOAT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch1f(attr, vertexID);
}

float nbl_glsl_VG_attribFetch_R16_SNORM(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch1f(attr, vertexID);
}

float nbl_glsl_VG_attribFetch_R16_UNORM(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch1f(attr, vertexID);
}

vec3 nbl_glsl_VG_attribFetch_RGB16_SFLOAT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch3f(attr, vertexID);
}

vec3 nbl_glsl_VG_attribFetch_RGB16_SNORM(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch3f(attr, vertexID);
}

vec3 nbl_glsl_VG_attribFetch_RGB16_UNORM(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch3f(attr, vertexID);
}
#endif

#if _NBL_VG_INT_BUFFERS_COUNT
int nbl_glsl_VG_attribFetch_R8_SINT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch1i(attr, vertexID);
}

ivec2 nbl_glsl_VG_attribFetch_RG8_SINT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch2i(attr, vertexID);
}

ivec3 nbl_glsl_VG_attribFetch_RGB8_SINT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch3i(attr, vertexID);
}

int nbl_glsl_VG_attribFetch_R16_SINT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch1i(attr, vertexID);
}

ivec2 nbl_glsl_VG_attribFetch_RG16_SINT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch2i(attr, vertexID);
}

ivec3 nbl_glsl_VG_attribFetch_RGB16_SINT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch3i(attr, vertexID);
}
#endif

#if _NBL_VG_UINT_BUFFERS_COUNT
uint nbl_glsl_VG_attribFetch_R8_UINT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch1u(attr, vertexID);
}

uvec2 nbl_glsl_VG_attribFetch_RG8_UINT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch2u(attr, vertexID);
}

uvec3 nbl_glsl_VG_attribFetch_RGB8_UINT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch3u(attr, vertexID);
}

uint nbl_glsl_VG_attribFetch_R16_UINT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch1u(attr, vertexID);
}

vec2 nbl_glsl_VG_attribFetch_RG16_UINT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch2u(attr, vertexID);
}

uvec3 nbl_glsl_VG_attribFetch_RGB16_UINT(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  return nbl_glsl_VG_attribFetch3u(attr, vertexID);
}
#endif

#endif