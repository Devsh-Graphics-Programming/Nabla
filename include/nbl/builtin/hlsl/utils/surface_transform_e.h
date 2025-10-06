
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SURFACE_TRANSFORM_E_INCLUDED_
#define _NBL_BUILTIN_HLSL_SURFACE_TRANSFORM_E_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace surface_transform
{


static const uint IDENTITY = 0x00000001;
static const uint ROTATE_90 = 0x00000002;
static const uint ROTATE_180 = 0x00000004;
static const uint ROTATE_270 = 0x00000008;
static const uint HORIZONTAL_MIRROR = 0x00000010;
static const uint HORIZONTAL_MIRROR_ROTATE_90 = 0x00000020;
static const uint HORIZONTAL_MIRROR_ROTATE_180 = 0x00000040;
static const uint HORIZONTAL_MIRROR_ROTATE_270 = 0x00000080;


}
}
}

#endif