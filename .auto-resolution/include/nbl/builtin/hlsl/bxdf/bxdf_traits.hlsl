// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_TRAITS_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_TRAITS_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{

enum BxDFType : uint16_t
{
    BT_BSDF = 0,
    BT_BRDF,
    BT_BTDF
};

template<typename BxdfT>
struct traits;

// defined individually after each bxdf definition

}
}
}

#endif
