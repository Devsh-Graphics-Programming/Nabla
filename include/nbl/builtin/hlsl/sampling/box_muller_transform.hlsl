// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_BOX_MULLER_TRANSFORM_INCLUDED_
#define _NBL_BUILTIN_HLSL_BOX_MULLER_TRANSFORM_INCLUDED_

#include "nbl/builtin/hlsl/math/functions.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"

namespace nbl
{
namespace hlsl
{

template<typename T>
vector<T,2> boxMullerTransform(vector<T,2> xi, T stddev)
{
    T sinPhi, cosPhi;
    nbl::hlsl::sincos<T>(2.0 * numbers::pi<float> * xi.y - numbers::pi<float>, sinPhi, cosPhi);
    return vector<T,2>(cosPhi, sinPhi) * nbl::hlsl::sqrt(-2.0 * nbl::hlsl::log(xi.x)) * stddev;
}

}
}

#endif
