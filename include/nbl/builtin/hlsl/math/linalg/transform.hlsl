// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MATH_LINALG_TRANSFORM_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_LINALG_TRANSFORM_INCLUDED_


#include <nbl/builtin/hlsl/mpl.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl>
#include <nbl/builtin/hlsl/concepts.hlsl>


namespace nbl
{
namespace hlsl
{
namespace math
{
namespace linalg
{

/// Builds a rotation 4 * 4 matrix created from an axis vector and an angle.
///
/// @param angle Rotation angle expressed in radians.
/// @param axis Rotation axis, must be normalized.
///
/// @tparam T A floating-point scalar type
template <typename T>
matrix<T, 3, 3> rotation_mat(T angle, vector<T, 3> const& axis)
{
  T const a = angle;
  T const c = cos(a);
  T const s = sin(a);

  vector<T, 3> temp((T(1) - c) * axis);

  matrix<T, 3, 3> rotation;
  rotation[0][0] = c + temp[0] * axis[0];
  rotation[0][1] = temp[1] * axis[0] - s * axis[2];
  rotation[0][2] = temp[2] * axis[0] + s * axis[1];

  rotation[1][0] = temp[0] * axis[1] + s * axis[2];
  rotation[1][1] = c + temp[1] * axis[1];
  rotation[1][2] = temp[2] * axis[1] - s * axis[0];

  rotation[2][0] = temp[0] * axis[2] - s * axis[1];
  rotation[2][1] = temp[1] * axis[2] + s * axis[0];
  rotation[2][2] = c + temp[2] * axis[2];

  return rotation;
}

}
}
}
}
#endif
