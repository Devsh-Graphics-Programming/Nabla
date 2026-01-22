// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SHAPES_SPHERICAL_RECTANGLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_SPHERICAL_RECTANGLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/numbers.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/math/angle_adding.hlsl>

namespace nbl
{
namespace hlsl
{
namespace shapes
{

template<typename Scalar>
struct SphericalRectangle
{
    using scalar_type = Scalar;
    using vector2_type = vector<Scalar, 2>;
    using vector3_type = vector<Scalar, 3>;
    using matrix3x3_type = matrix<Scalar, 3, 3>;

    static SphericalRectangle<Scalar> create(const vector3_type rectangleOrigin, const vector3_type right, const vector3_type up)
    {
        SphericalRectangle<scalar_type> retval;
        retval.origin = rectangleOrigin;
        retval.extents = vector2_type(hlsl::length(right), hlsl::length(up));
        retval.basis[0] = right / retval.extents[0];
        retval.basis[1] = up / retval.extents[1];
        retval.basis[2] = hlsl::normalize(hlsl::cross(retval.basis[0], retval.basis[1]));
        return retval;
    }

    scalar_type solidAngle(const vector3_type observer)
    {
        const vector3_type r0 = hlsl::mul(basis, origin - observer);

        using vector4_type = vector<Scalar, 4>;
        const vector4_type denorm_n_z = vector4_type(-r0.y, r0.x + extents.x, r0.y + extents.y, -r0.x);
        const vector4_type n_z = denorm_n_z / nbl::hlsl::sqrt((vector4_type)(r0.z * r0.z) + denorm_n_z * denorm_n_z);
        const vector4_type cosGamma = vector4_type(
            -n_z[0] * n_z[1],
            -n_z[1] * n_z[2],
            -n_z[2] * n_z[3],
            -n_z[3] * n_z[0]
        );
        math::sincos_accumulator<scalar_type> angle_adder = math::sincos_accumulator<scalar_type>::create(cosGamma[0]);
        angle_adder.addCosine(cosGamma[1]);
        angle_adder.addCosine(cosGamma[2]);
        angle_adder.addCosine(cosGamma[3]);
        return angle_adder.getSumofArccos() - scalar_type(2.0) * numbers::pi<float>;
    }

    vector3_type origin;
    vector2_type extents;
    matrix3x3_type basis;
};

}
}
}

#endif
