// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_VIRTUAL_GEOMETRY_DESCRIPTORS_INCLUDED_
#define _NBL_BUILTIN_GLSL_VIRTUAL_GEOMETRY_DESCRIPTORS_INCLUDED_



#ifndef _NBL_VG_USE_SSBO


#ifndef _NBL_VG_DESCRIPTOR_SET
#define _NBL_VG_DESCRIPTOR_SET 0
#endif

#ifndef _NBL_VG_UINT_BUFFERS
#define _NBL_VG_UINT_BUFFERS_BINDING 0
#define _NBL_VG_UINT_BUFFERS_COUNT 1
#endif
#ifndef _NBL_VG_FLOAT_BUFFERS
#define _NBL_VG_FLOAT_BUFFERS_BINDING 1 
#define _NBL_VG_FLOAT_BUFFERS_COUNT 4
#endif
#ifndef _NBL_VG_INT_BUFFERS
#define _NBL_VG_INT_BUFFERS_BINDING 2
#define _NBL_VG_INT_BUFFERS_COUNT 0
#endif

#if _NBL_VG_UINT_BUFFERS_COUNT
layout(set = _NBL_VG_DESCRIPTOR_SET, binding = _NBL_VG_UINT_BUFFERS_BINDING) uniform usamplerBuffer MeshPackedDataUintSample[_NBL_VG_UINT_BUFFERS_COUNT];
uint nbl_glsl_VG_fetchTriangleVertexIndex(in uint baseVertex, in uint triangleVx)
{
    return texelFetch(MeshPackedDataUintSample[_NBL_VG_UINT_BUFFERS_COUNT-1u],int(baseVertex+triangleVx)).x;
}
#endif
#if _NBL_VG_FLOAT_BUFFERS_COUNT
layout(set = _NBL_VG_DESCRIPTOR_SET, binding = _NBL_VG_FLOAT_BUFFERS_BINDING) uniform samplerBuffer MeshPackedDataFloatSample[_NBL_VG_FLOAT_BUFFERS_COUNT];
#endif
#if _NBL_VG_INT_BUFFERS_COUNT
layout(set = _NBL_VG_DESCRIPTOR_SET, binding = _NBL_VG_INT_BUFFERS_BINDING) uniform isamplerBuffer MeshPackedDataIntSample[_NBL_VG_INT_BUFFERS_COUNT];
#endif


#else // _NBL_VG_USE_SSBO


#ifndef _NBL_VG_SSBO_DESCRIPTOR_SET
#define _NBL_VG_SSBO_DESCRIPTOR_SET 0
#endif

#ifndef _NBL_VG_USE_SSBO_UINT
#ifndef _NBL_VG_SSBO_UINT_BINDING
#define _NBL_VG_SSBO_UINT_BINDING 0
#endif
#endif
#ifndef _NBL_VG_USE_SSBO_UVEC2
#ifndef _NBL_VG_SSBO_UVEC2_BINDING
#define _NBL_VG_SSBO_UVEC2_BINDING 1
#endif
#endif
#ifndef _NBL_VG_USE_SSBO_UVEC3
#ifndef _NBL_VG_SSBO_UVEC3_BINDING
#define _NBL_VG_SSBO_UVEC3_BINDING 2
#endif
#endif
#ifndef _NBL_VG_USE_SSBO_UVEC4
#ifndef _NBL_VG_SSBO_UVEC4_BINDING
#define _NBL_VG_SSBO_UVEC4_BINDING 3
#endif
#endif
#ifndef _NBL_VG_USE_SSBO_INDEX
#ifndef _NBL_VG_SSBO_INDEX_BINDING
#define _NBL_VG_SSBO_INDEX_BINDING 4
#endif
#endif

#ifdef _NBL_VG_USE_SSBO_UINT
layout(set = _NBL_VG_SSBO_DESCRIPTOR_SET, binding = _NBL_VG_SSBO_UINT_BINDING, std430) readonly buffer MeshPackedDataAsUint
{
    uint attribData[];
} meshPackedDataUintBuffer;
#endif
#ifdef _NBL_VG_USE_SSBO_UVEC2
layout(set = _NBL_VG_SSBO_DESCRIPTOR_SET, binding = _NBL_VG_SSBO_UVEC2_BINDING, std430) readonly buffer MeshPackedDataAsUvec2
{
    uvec2 attribData[];
} meshPackedDataUvec2Buffer;
#endif
#ifdef _NBL_VG_USE_SSBO_UVEC3
struct Packed_uvec3_t
{
    uint x, y, z;
};
layout(set = _NBL_VG_SSBO_DESCRIPTOR_SET, binding = _NBL_VG_SSBO_UVEC3_BINDING, std430) readonly buffer MeshPackedDataAsUvec3
{
    Packed_uvec3_t attribData[];
} meshPackedDataUvec3Buffer;
#endif
#ifdef _NBL_VG_USE_SSBO_UVEC4
layout(set = _NBL_VG_SSBO_DESCRIPTOR_SET, binding = _NBL_VG_SSBO_UVEC4_BINDING, std430) readonly buffer MeshPackedDataAsUvec4
{
    uvec4 attribData[];
} meshPackedDataUvec4Buffer;
#endif
#ifdef _NBL_VG_USE_SSBO_INDEX
layout(set = _NBL_VG_SSBO_DESCRIPTOR_SET, binding = _NBL_VG_SSBO_INDEX_BINDING, std430) readonly buffer TrianglePackedData
{
    uint indices[];
} trianglePackedData;
uint nbl_glsl_VG_fetchTriangleVertexIndex(in uint baseVertex, in uint triangleVx)
{
    uint realIndex = baseVertex + triangleVx;
    uint packedData = trianglePackedData.indices[realIndex>>1u];
    uint extractedData = uint(bitfieldExtract(packedData, bool(realIndex&0x1u) ? 16 : 0, 16));
    return extractedData;
}
#endif


#endif // _NBL_VG_USE_SSBO



#endif // _NBL_BUILTIN_GLSL_VIRTUAL_GEOMETRY_DESCRIPTORS_INCLUDED_