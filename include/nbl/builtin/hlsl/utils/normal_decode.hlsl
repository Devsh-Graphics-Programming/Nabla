
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_UTILS_NORMAL_DECODE_INCLUDED_
#define _NBL_BUILTIN_HLSL_UTILS_NORMAL_DECODE_INCLUDED_

#include "nbl/builtin/hlsl/math/constants.hlsl"

#include "nbl/builtin/hlsl/utils/compressed_normal_matrix_t.hlsl"


namespace nbl
{
namespace hlsl
{
namespace normal_decode
{


float3 signedSpherical(in float2 enc)
{
	float ang = enc.x * math::PI;
    return float3(float2(cos(ang),sin(ang))*sqrt(1.0-enc.y*enc.y), enc.y);
}


}
}
}

#endif