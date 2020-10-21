// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _IRR_BUILTIN_GLSL_VIRTUAL_TEXTURING_DESCRIPTORS_INCLUDED_
#define _IRR_BUILTIN_GLSL_VIRTUAL_TEXTURING_DESCRIPTORS_INCLUDED_

#define _IRR_VT_MAX_PAGE_TABLE_LAYERS 256

#ifndef _IRR_VT_DESCRIPTOR_SET
#define _IRR_VT_DESCRIPTOR_SET 0
#endif
#ifndef _IRR_VT_PAGE_TABLE_BINDING
#define _IRR_VT_PAGE_TABLE_BINDING 0
#endif
#ifndef _IRR_VT_FLOAT_VIEWS
#define _IRR_VT_FLOAT_VIEWS_BINDING 1 
#define _IRR_VT_FLOAT_VIEWS_COUNT 15
#endif
#ifndef _IRR_VT_INT_VIEWS
#define _IRR_VT_INT_VIEWS_BINDING 2
#define _IRR_VT_INT_VIEWS_COUNT 0
#endif
#ifndef _IRR_VT_UINT_VIEWS
#define _IRR_VT_UINT_VIEWS_BINDING 3
#define _IRR_VT_UINT_VIEWS_COUNT 0
#endif

layout(set = _IRR_VT_DESCRIPTOR_SET, binding = _IRR_VT_PAGE_TABLE_BINDING) uniform usampler2DArray pageTable;
#if _IRR_VT_FLOAT_VIEWS_COUNT
layout(set = _IRR_VT_DESCRIPTOR_SET, binding = _IRR_VT_FLOAT_VIEWS_BINDING) uniform sampler2DArray physicalTileStorageFormatView[_IRR_VT_FLOAT_VIEWS_COUNT];
#endif
#if _IRR_VT_INT_VIEWS_COUNT
layout(set = _IRR_VT_DESCRIPTOR_SET, binding = _IRR_VT_INT_VIEWS_BINDING) uniform isampler2DArray iphysicalTileStorageFormatView[_IRR_VT_INT_VIEWS_COUNT];
#endif
#if _IRR_VT_UINT_VIEWS_COUNT
layout(set = _IRR_VT_DESCRIPTOR_SET, binding = _IRR_VT_UINT_VIEWS_BINDING) uniform usampler2DArray uphysicalTileStorageFormatView[_IRR_VT_UINT_VIEWS_COUNT];
#endif

#endif