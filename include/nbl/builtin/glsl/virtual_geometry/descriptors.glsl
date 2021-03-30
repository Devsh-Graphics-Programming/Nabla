// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_VIRTUAL_GEOMETRY_DESCRIPTORS_INCLUDED_
#define _NBL_BUILTIN_GLSL_VIRTUAL_GEOMETRY_DESCRIPTORS_INCLUDED_

#ifndef _NBL_VG_DESCRIPTOR_SET
#define _NBL_VG_DESCRIPTOR_SET 0
#endif
#ifndef _NBL_VG_FLOAT_BUFFERS
#define _NBL_VG_FLOAT_BUFFERS_BINDING 1 
#define _NBL_VG_FLOAT_BUFFERS_COUNT 15
#endif
#ifndef _NBL_VG_INT_BUFFERS
#define _NBL_VG_INT_BUFFERS_BINDING 2
#define _NBL_VG_INT_BUFFERS_COUNT 0
#endif
#ifndef _NBL_VG_UINT_BUFFERS
#define _NBL_VG_UINT_BUFFERS_BINDING 3
#define _NBL_VG_UINT_BUFFERS_COUNT 0
#endif

#if _NBL_VG_FLOAT_BUFFERS_COUNT
layout(set = _NBL_VG_DESCRIPTOR_SET, binding = _NBL_VG_FLOAT_BUFFERS_BINDING) uniform samplerBuffer MeshPackedDataFloatSample[_NBL_VG_FLOAT_BUFFERS_COUNT];
#endif
#if _NBL_VG_INT_BUFFERS_COUNT
layout(set = _NBL_VG_DESCRIPTOR_SET, binding = _NBL_VG_INT_BUFFERS_BINDING) uniform isamplerBuffer MeshPackedDataIntSample[_NBL_VG_INT_BUFFERS_COUNT];
#endif
#if _NBL_VG_UINT_BUFFERS_COUNT
layout(set = _NBL_VG_DESCRIPTOR_SET, binding = _NBL_VG_UINT_BUFFERS_BINDING) uniform usamplerBuffer MeshPackedDataUintSample[_NBL_VG_UINT_BUFFERS_COUNT];
#endif

#endif