// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SHAPES_TRIANGLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_TRIANGLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{
namespace shapes
{

namespace util
{
  template <typename float_t>
  vector<float_t, 3> GetAngleWeight(const vector<float_t, 3>& e1, const vector<float_t, 3>& e2, const vector<float_t, 3>& e3)
  {
    // Calculate this triangle's weight for each of its three m_vertices
    // start by calculating the lengths of its sides
    const float_t a = dot(e1, e1);
    const float_t asqrt = sqrt(a);
    const float_t b = dot(e2, e2);
    const float_t bsqrt = sqrt(b);
    const float_t c = dot(e3, e3);
    const float_t csqrt = sqrt(c);

    // use them to find the angle at each vertex
    return vector<float_t, 3>(
      acosf((b + c - a) / (2.f * bsqrt * csqrt)),
      acosf((-b + c + a) / (2.f * asqrt * csqrt)),
      acosf((b - c + a) / (2.f * bsqrt * asqrt)));
  }
}

}
}
}

#endif
