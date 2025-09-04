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

template <uint16_t NOut, uint16_t MOut, uint16_t NIn, uint16_t MIn, typename T>
requires(NOut >= NIn && MOut >= MIn)
matrix <T, NOut, MOut> promote_affine(const matrix<T, NIn, MIn> inMatrix)
{
  matrix<T, NOut, MOut> retval;

  using out_row_t = hlsl::vector<T, MOut>;
  auto expandVec = [](vector<T, MIn> inVec) -> vector<T, MOut>
  {
    if constexpr (MIn == MOut) return inVec;
    return vector<T, MOut>(inVec, vector<T, MOut - MIn>(0));
  };

  for (auto row_i = 0u; row_i < NOut; row_i++)
  {
    retval[row_i] = row_i < NIn ? expandVec(inMatrix[row_i]) : promote<out_row_t>(0.0f);
    if ((row_i >= NIn || row_i >= MIn) && row_i < MOut) retval[row_i][row_i] = T(1);
    
  }
  return retval;
}

}
}
}
}
#endif
