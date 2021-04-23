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

#if _NBL_VG_FLOAT_BUFFERS_COUNT

vec4 nbl_glsl_VG_attribFetch4f(in nbl_glsl_VG_VirtualAttributePacked_t attr, uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
  const int addr = va.offset+int(vertexID);
#ifdef NBL_GLSL_VG_USE_SSBO
  return MeshPackedDataVec4Buffer[va.binding][addr];
#else
  return texelFetch(MeshPackedDataFloatSample[va.binding],addr);
#endif
}
vec3 nbl_glsl_VG_attribFetch3f(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
  const int addr = va.offset+int(vertexID);
#if _NBL_VG_USE_SSBO
  return meshPackedDataUvec3Buffer[va];
#else
  return texelFetch(MeshPackedDataFloatSample[va.binding],addr).xyz;
#endif
}
vec2 nbl_glsl_VG_attribFetch2f(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
  const int addr = va.offset+int(vertexID);
#if _NBL_VG_USE_SSBO
  return MeshPackedDataVec2Buffer[va];
#else
  return texelFetch(MeshPackedDataFloatSample[va.binding],addr).xy;
#endif
}
float nbl_glsl_VG_attribFetch1f(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
  const int addr = va.offset+int(vertexID);
#if NBL_GLSL_VG_USE_SSBO
  return meshPackedDataUvec2Buffer[va.binding][addr];
#else
  return texelFetch(MeshPackedDataFloatSample[va.binding],addr).x;
#endif
}

#endif // _NBL_VG_FLOAT_BUFFERS_COUNT

#if _NBL_VG_INT_BUFFERS_COUNT

ivec4 nbl_glsl_VG_attribFetch4i(in nbl_glsl_VG_VirtualAttributePacked_t attr, uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
  const int addr = va.offset+int(vertexID);
#if NBL_GLSL_VG_USE_SSBO
  return MeshPackedDataIvec4Buffer[va.binding][addr];
#else
  return texelFetch(MeshPackedDataIntSample[va.binding],addr);
#endif
}
ivec3 nbl_glsl_VG_attribFetch3i(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
  const int addr = va.offset+int(vertexID);
#if NBL_GLSL_VG_USE_SSBO
  return MeshPackedDataIvec3Buffer[va.binding][addr];
#else
  return texelFetch(MeshPackedDataIntSample[va.binding],addr).xyz;
#endif
}
ivec2 nbl_glsl_VG_attribFetch2i(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
  const int addr = va.offset+int(vertexID);
#if NBL_GLSL_VG_USE_SSBO
  return MeshPackedDataIvec2Buffer[va.binding][addr];
#else
  return texelFetch(MeshPackedDataIntSample[va.binding],addr).xy;
#endif
}
int nbl_glsl_VG_attribFetch1i(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
  const int addr = va.offset+int(vertexID);
#if NBL_GLSL_VG_USE_SSBO
  return MeshPackedDataIntBuffer[va.binding][addr];
#else
  return texelFetch(MeshPackedDataIntSample[va.binding],addr).x;
#endif
}

#endif // _NBL_VG_INT_BUFFERS_COUNT

#if _NBL_VG_UINT_BUFFERS_COUNT

uvec4 nbl_glsl_VG_attribFetch4u(in nbl_glsl_VG_VirtualAttributePacked_t attr, uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
  const int addr = va.offset+int(vertexID);
#if NBL_GLSL_VG_USE_SSBO
  return MeshPackedDataUvec4Buffer[va.binding][addr];
#else
  return texelFetch(MeshPackedDataUintSample[va.binding],addr);
#endif
}
uvec3 nbl_glsl_VG_attribFetch3u(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
  const int addr = va.offset+int(vertexID);
#if NBL_GLSL_VG_USE_SSBO
  return MeshPackedDataUvec3Buffer[va.binding][addr];
#else
  return texelFetch(MeshPackedDataUintSample[va.binding],addr).xyz;
#endif
}
uvec2 nbl_glsl_VG_attribFetch2u(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
  const int addr = va.offset+int(vertexID);
#if NBL_GLSL_VG_USE_SSBO
  return MeshPackedDataUvec2Buffer[va.binding][addr];
#else
  return texelFetch(MeshPackedDataUintSample[va.binding],addr).xy;
#endif
}
uint nbl_glsl_VG_attribFetch1u(in nbl_glsl_VG_VirtualAttributePacked_t attr, in uint vertexID)
{
  const nbl_glsl_VG_VirtualAttribute va = nbl_glsl_VG_unpackVirtualAttribute(attr);
  const int addr = va.offset+int(vertexID);
#if _NBL_VG_USE_SSBO
  return meshPackedDataUintBuffer[va];
#else
  return texelFetch(MeshPackedDataUintSample[va.binding],addr).x;
#endif
}

#endif // _NBL_VG_UINT_BUFFERS_COUNT

#endif