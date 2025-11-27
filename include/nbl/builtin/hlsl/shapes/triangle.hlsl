// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SHAPES_TRIANGLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_TRIANGLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/tgmath.hlsl>
#include <nbl/builtin/hlsl/numbers.hlsl>

namespace nbl
{
namespace hlsl
{
namespace shapes
{

namespace util
{
  // Use this convetion e_i = v_{i+2}-v_{i+1}. vertex index is modulo by 3.
  template <typename float_t>
  vector<float_t, 3> compInternalAngle(NBL_CONST_REF_ARG(vector<float_t, 3>) e0, NBL_CONST_REF_ARG(vector<float_t, 3>) e1, NBL_CONST_REF_ARG(vector<float_t, 3>) e2)
  {
    // Calculate this triangle's weight for each of its three m_vertices
    // start by calculating the lengths of its sides
    const float_t a = hlsl::dot(e0, e0);
    const float_t asqrt = hlsl::sqrt(a);
    const float_t b = hlsl::dot(e1, e1);
    const float_t bsqrt = hlsl::sqrt(b);
    const float_t c = hlsl::dot(e2, e2);
    const float_t csqrt = hlsl::sqrt(c);

    const float_t angle0 = hlsl::acos((b + c - a) / (2.f * bsqrt * csqrt));
    const float_t angle1 = hlsl::acos((-b + c + a) / (2.f * asqrt * csqrt));
    const float_t angle2 = hlsl::numbers::pi<float_t> - (angle0 + angle1);
    // use them to find the angle at each vertex
    return vector<float_t, 3>(angle0, angle1, angle2);
  }
}

}
}
}

#endif
