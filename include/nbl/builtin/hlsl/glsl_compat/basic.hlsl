// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_GLSL_COMPAT_BASIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_GLSL_COMPAT_BASIC_INCLUDED_

#include "nbl/builtin/hlsl/spirv_intrinsics/basic.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace glsl
{
	
void barrier() {
	spirv::controlBarrier(2, 2, 0x8 | 0x100);
}

void memoryBarrierShared() {
	spirv::memoryBarrier(1, 0x8 | 0x100);
}

}
}
}

#endif