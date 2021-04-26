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
    result.binding = vaPacked >> 28;
    result.offset = int(vaPacked & 0x0FFFFFFF);
    
    return result;
#else
    return vaPacked & 0x0FFFFFFF;
#endif
}

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC4)) || defined(_NBL_VG_FLOAT_BUFFERS_COUNT)
vec4 nbl_glsl_VG_attribFetch4f(in nbl_glsl_VG_VirtualAttributePacked_t attr, uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
#ifdef _NBL_VG_USE_SSBO
  uvec4 attrLocal = meshPackedDataUvec4Buffer.attribData[va + vertexID];
  return vec4(uintBitsToFloat(attrLocal.x), uintBitsToFloat(attrLocal.y), uintBitsToFloat(attrLocal.z), uintBitsToFloat(attrLocal.w));
#else
  const int addr = va.offset+int(vertexID);
  return texelFetch(MeshPackedDataFloatSample[va.binding],addr);
#endif
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC3)) || _NBL_VG_FLOAT_BUFFERS_COUNT
vec3 nbl_glsl_VG_attribFetch3f(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
#ifdef _NBL_VG_USE_SSBO
  Packed_uvec3_t attrLocal = meshPackedDataUvec3Buffer.attribData[va + vertexID];
  return vec3(uintBitsToFloat(attrLocal.x), uintBitsToFloat(attrLocal.y), uintBitsToFloat(attrLocal.z));
#else
  const int addr = va.offset+int(vertexID);
  return texelFetch(MeshPackedDataFloatSample[va.binding],addr).xyz;
#endif
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC2)) || _NBL_VG_FLOAT_BUFFERS_COUNT
vec2 nbl_glsl_VG_attribFetch2f(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
#ifdef _NBL_VG_USE_SSBO
  uvec2 attrLocal = meshPackedDataUvec2Buffer.attribData[va + vertexID];
  return vec2(uintBitsToFloat(attrLocal.x), uintBitsToFloat(attrLocal.y));
#else
  const int addr = va.offset+int(vertexID);
  return texelFetch(MeshPackedDataFloatSample[va.binding],addr).xy;
#endif
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UINT)) || _NBL_VG_FLOAT_BUFFERS_COUNT
float nbl_glsl_VG_attribFetch1f(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
#ifdef _NBL_VG_USE_SSBO
  return uintBitsToFloat(meshPackedDataUintBuffer.attribData[va + vertexID]);
#else
  const int addr = va.offset+int(vertexID);
  return texelFetch(MeshPackedDataFloatSample[va.binding],addr).x;
#endif
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC4)) || _NBL_VG_INT_BUFFERS_COUNT
ivec4 nbl_glsl_VG_attribFetch4i(in nbl_glsl_VG_VirtualAttributePacked_t attr, uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
#ifdef _NBL_VG_USE_SSBO
  return ivec4(meshPackedDataUvec4Buffer.attribData[va + vertexID]);
#else
  const int addr = va.offset+int(vertexID);
  return texelFetch(MeshPackedDataIntSample[va.binding],addr);
#endif
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC3)) || _NBL_VG_INT_BUFFERS_COUNT
ivec3 nbl_glsl_VG_attribFetch3i(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
#ifdef _NBL_VG_USE_SSBO
  Packed_uvec3_t attrLocal = meshPackedDataUvec3Buffer.attribData[va + vertexID];
  return ivec3(int(attrLocal.x), int(attrLocal.y), int(attrLocal.z));
#else
  const int addr = va.offset+int(vertexID);
  return texelFetch(MeshPackedDataIntSample[va.binding],addr).xyz;
#endif
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC2)) || _NBL_VG_INT_BUFFERS_COUNT
ivec2 nbl_glsl_VG_attribFetch2i(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
#ifdef _NBL_VG_USE_SSBO
  return ivec2(meshPackedDataUvec2Buffer.attribData[va + vertexID]);
#else
  const int addr = va.offset+int(vertexID);
  return texelFetch(MeshPackedDataIntSample[va.binding],addr).xy;
#endif
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UINT)) || _NBL_VG_INT_BUFFERS_COUNT
int nbl_glsl_VG_attribFetch1i(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
#ifdef _NBL_VG_USE_SSBO
  return int(meshPackedDataUintBuffer.attribData[va + vertexID]);
#else
  const int addr = va.offset+int(vertexID);
  return texelFetch(MeshPackedDataIntSample[va.binding],addr).x;
#endif
}
#endif

#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC4)) || _NBL_VG_UINT_BUFFERS_COUNT
uvec4 nbl_glsl_VG_attribFetch4u(in nbl_glsl_VG_VirtualAttributePacked_t attr, uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
#ifdef _NBL_VG_USE_SSBO
  return meshPackedDataUvec4Buffer.attribData[va + vertexID];
#else
  const int addr = va.offset+int(vertexID);
  return texelFetch(MeshPackedDataUintSample[va.binding],addr);
#endif
}
#endif
#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC3)) || _NBL_VG_UINT_BUFFERS_COUNT
uvec3 nbl_glsl_VG_attribFetch3u(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
#ifdef _NBL_VG_USE_SSBO
  Packed_uvec3_t attrLocal = meshPackedDataUvec3Buffer.attribData[va + vertexID];
  return uvec3(attrLocal.x, attrLocal.y, attrLocal.z);
#else
  const int addr = va.offset+int(vertexID);
  return texelFetch(MeshPackedDataUintSample[va.binding],addr).xyz;
#endif
}
#endif
#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UVEC2)) || _NBL_VG_UINT_BUFFERS_COUNT
uvec2 nbl_glsl_VG_attribFetch2u(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
#ifdef _NBL_VG_USE_SSBO
  return meshPackedDataUvec2Buffer.attribData[va + vertexID];
#else
  const int addr = va.offset+int(vertexID);
  return texelFetch(MeshPackedDataUintSample[va.binding],addr).xy;
#endif
}
#endif
#if (defined(_NBL_VG_USE_SSBO) && defined(_NBL_VG_USE_SSBO_UINT)) || _NBL_VG_UINT_BUFFERS_COUNT
uint nbl_glsl_VG_attribFetch1u(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
#ifdef _NBL_VG_USE_SSBO
  return meshPackedDataUintBuffer.attribData[va + vertexID];
#else
  const int addr = va.offset+int(vertexID);
  return texelFetch(MeshPackedDataUintSample[va.binding],addr).x;
#endif
}
#endif

#endif