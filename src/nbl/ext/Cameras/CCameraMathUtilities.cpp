// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/Cameras/CCameraMathUtilities.hpp"

namespace nbl::hlsl
{

namespace
{

template<typename T>
camera_quaternion_t<T> makeQuaternionFromBasisWithCast(
    const camera_vector_t<T, 3>& right,
    const camera_vector_t<T, 3>& up,
    const camera_vector_t<T, 3>& forward)
{
    const camera_matrix_t<T, 3, 3> basis(right, up, forward);
    const auto candidate = _static_cast<camera_quaternion_t<T>>(basis);
    if (!CCameraMathUtilities::isFiniteQuaternion(candidate))
        return CCameraMathUtilities::makeIdentityQuaternion<T>();

    return CCameraMathUtilities::normalizeQuaternion(candidate);
}

} // namespace

camera_quaternion_t<float> CCameraMathUtilities::makeQuaternionFromBasisImpl(
    const camera_vector_t<float, 3>& right,
    const camera_vector_t<float, 3>& up,
    const camera_vector_t<float, 3>& forward)
{
    return makeQuaternionFromBasisWithCast(right, up, forward);
}

camera_quaternion_t<double> CCameraMathUtilities::makeQuaternionFromBasisImpl(
    const camera_vector_t<double, 3>& right,
    const camera_vector_t<double, 3>& up,
    const camera_vector_t<double, 3>& forward)
{
    return makeQuaternionFromBasisWithCast(right, up, forward);
}

} // namespace nbl::hlsl
