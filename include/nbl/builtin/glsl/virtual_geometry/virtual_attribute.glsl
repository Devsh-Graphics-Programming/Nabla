// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_VIRTUAL_GEOMETRY_VIRTUAL_ATTRIBUTE_INCLUDED_
#define _NBL_BUILTIN_GLSL_VIRTUAL_GEOMETRY_VIRTUAL_ATTRIBUTE_INCLUDED_

#include <nbl/builtin/glsl/virtual_geometry/descriptors.glsl>

#define nbl_glsl_VG_VirtualAttributePacked_t uint

#ifndef _NBL_VG_USE_SSBO
struct nbl_glsl_VG_VirtualAttribute
{
    uint binding;
    int offset;
};
#else
#define nbl_glsl_VG_VirtualAttribute uint
#endif

nbl_glsl_VG_VirtualAttribute nbl_glsl_VG_unpackVirtualAttribute(in nbl_glsl_VG_VirtualAttributePacked_t vaPacked)
{
#ifndef _NBL_VG_USE_SSBO
    nbl_glsl_VG_VirtualAttribute result;
    result.binding = bitfieldExtract(vaPacked, 0, 4);
    result.offset = int(bitfieldExtract(vaPacked, 4, 28));
    
    return result;
#else
    return vaPacked & 0x0FFFFFFF;
#endif
}

#if defined(_NBL_VG_FLOAT_BUFFERS_COUNT) || defined(_NBL_VG_USE_SSBO)

#if defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC4)
vec4 nbl_glsl_VG_attribFetch4f(in nbl_glsl_VG_VirtualAttributePacked_t attr, uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
#ifdef _NBL_VG_USE_SSBO
  return vec4(meshPackedDataUvec4Buffer.attribData[va]);
#else
  const int addr = va.offset+int(vertexID);
  return texelFetch(MeshPackedDataFloatSample[va.binding],addr);
#endif
}
#endif
#if defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC3)
vec3 nbl_glsl_VG_attribFetch3f(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
#ifdef _NBL_VG_USE_SSBO
  return vec3(meshPackedDataUvec3Buffer.attribData[va]);
#else
  const int addr = va.offset+int(vertexID);
  return texelFetch(MeshPackedDataFloatSample[va.binding],addr).xyz;
#endif
}
#endif
#if defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC2)
vec2 nbl_glsl_VG_attribFetch2f(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
#ifdef _NBL_VG_USE_SSBO
  return vec2(meshPackedDataUvec2Buffer.attribData[va]);
#else
  const int addr = va.offset+int(vertexID);
  return texelFetch(MeshPackedDataFloatSample[va.binding],addr).xy;
#endif
}
#endif
#if defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UINT)
float nbl_glsl_VG_attribFetch1f(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
#ifdef _NBL_VG_USE_SSBO
  return float(meshPackedDataUintBuffer.attribData[va]);
#else
  const int addr = va.offset+int(vertexID);
  return texelFetch(MeshPackedDataFloatSample[va.binding],addr).x;
#endif
}
#endif

#endif
#if defined(_NBL_VG_INT_BUFFERS_COUNT) || defined(_NBL_VG_USE_SSBO)

#if defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC4)
ivec4 nbl_glsl_VG_attribFetch4i(in nbl_glsl_VG_VirtualAttributePacked_t attr, uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
#ifdef _NBL_VG_USE_SSBO
  return ivec4(meshPackedDataUvec4Buffer.attribData[va]);
#else
  const int addr = va.offset+int(vertexID);
  return texelFetch(MeshPackedDataIntSample[va.binding],addr);
#endif
}
#endif
#if defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC3)
ivec3 nbl_glsl_VG_attribFetch3i(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
#ifdef _NBL_VG_USE_SSBO
  return ivec3(meshPackedDataUvec3Buffer.attribData[va]);
#else
  const int addr = va.offset+int(vertexID);
  return texelFetch(MeshPackedDataIntSample[va.binding],addr).xyz;
#endif
}
#endif
#if defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC2)
ivec2 nbl_glsl_VG_attribFetch2i(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
#ifdef _NBL_VG_USE_SSBO
  return ivec2(meshPackedDataUvec2Buffer.attribData[va]);
#else
  const int addr = va.offset+int(vertexID);
  return texelFetch(MeshPackedDataIntSample[va.binding],addr).xy;
#endif
}
#endif
#if defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UINT)
int nbl_glsl_VG_attribFetch1i(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
#ifdef _NBL_VG_USE_SSBO
  return int(meshPackedDataUintBuffer.attribData[va]);
#else
  const int addr = va.offset+int(vertexID);
  return texelFetch(MeshPackedDataIntSample[va.binding],addr).x;
#endif
}
#endif

#endif
#if defined(_NBL_VG_UINT_BUFFERS_COUNT) || defined(_NBL_VG_USE_SSBO)

#if defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC4)
uvec4 nbl_glsl_VG_attribFetch4u(in nbl_glsl_VG_VirtualAttributePacked_t attr, uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
#ifdef _NBL_VG_USE_SSBO
  return meshPackedDataUvec4Buffer.attribData[va];
#else
  const int addr = va.offset+int(vertexID);
  return texelFetch(MeshPackedDataUintSample[va.binding],addr);
#endif
}
#endif
#if defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC3)
uvec3 nbl_glsl_VG_attribFetch3u(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
#ifdef _NBL_VG_USE_SSBO
  return meshPackedDataUvec3Buffer.attribData[va];
#else
  const int addr = va.offset+int(vertexID);
  return texelFetch(MeshPackedDataUintSample[va.binding],addr).xyz;
#endif
}
#endif
#if defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC2)
uvec2 nbl_glsl_VG_attribFetch2u(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
#ifdef _NBL_VG_USE_SSBO
  return meshPackedDataUvec2Buffer.attribData[va];
#else
  const int addr = va.offset+int(vertexID);
  return texelFetch(MeshPackedDataUintSample[va.binding],addr).xyz;
#endif
}
#endif
#if defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UINT)
uint nbl_glsl_VG_attribFetch1u(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
#ifdef _NBL_VG_USE_SSBO
  return meshPackedDataUintBuffer.attribData[va];
#else
  const int addr = va.offset+int(vertexID);
  return texelFetch(MeshPackedDataUintSample[va.binding],addr).xyz;
#endif
}
#endif

#endif
#endif