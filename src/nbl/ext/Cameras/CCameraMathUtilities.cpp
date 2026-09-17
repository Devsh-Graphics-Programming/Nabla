// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/Cameras/CCameraMathUtilities.hpp"

namespace nbl::hlsl
{

namespace
{

template<typename T>
math::quaternion<T> makeQuaternionFromBasisWithCast(
    const vector<T, 3>& right,
    const vector<T, 3>& up,
    const vector<T, 3>& forward)
{
    const matrix<T, 3, 3> basis(right, up, forward);
    const auto candidate = _static_cast<math::quaternion<T>>(basis);
    if (!CCameraMathUtilities::isFiniteQuaternion(candidate))
        return CCameraMathUtilities::makeIdentityQuaternion<T>();

    return CCameraMathUtilities::normalizeQuaternion(candidate);
}

} // namespace

math::quaternion<float> CCameraMathUtilities::makeQuaternionFromBasisImpl(
    const vector<float, 3>& right,
    const vector<float, 3>& up,
    const vector<float, 3>& forward)
{
    return makeQuaternionFromBasisWithCast(right, up, forward);
}

math::quaternion<double> CCameraMathUtilities::makeQuaternionFromBasisImpl(
    const vector<double, 3>& right,
    const vector<double, 3>& up,
    const vector<double, 3>& forward)
{
    return makeQuaternionFromBasisWithCast(right, up, forward);
}

} // namespace nbl::hlsl
