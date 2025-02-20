// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SHAPES_RECTANGLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_RECTANGLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/numbers.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>

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
    using vector3_type = vector<Scalar, 3>;
    using vector4_type = vector<Scalar, 4>;
    using matrix3x3_type = matrix<Scalar, 3, 3>;

    static SphericalRectangle<scalar_type> create(NBL_CONST_REF_ARG(vector3_type) observer, NBL_CONST_REF_ARG(vector3_type) rectangleOrigin, NBL_CONST_REF_ARG(matrix3x3_type) basis)
    {
        SphericalRectangle<scalar_type> retval;
        retval.r0 = nbl::hlsl::mul(basis, rectangleOrigin - observer);
        return retval;
    }

    static SphericalRectangle<Scalar> create(NBL_CONST_REF_ARG(vector3_type) observer, NBL_CONST_REF_ARG(vector3_type) rectangleOrigin, NBL_CONST_REF_ARG(vector3_type) T, NBL_CONST_REF_ARG(vector3_type) B, NBL_CONST_REF_ARG(vector3_type) N)
    {
        SphericalRectangle<scalar_type> retval;
        matrix3x3_type TBN = nbl::hlsl::transpose<matrix3x3_type>(matrix3x3_type(T, B, N));
        retval.r0 = nbl::hlsl::mul(TBN, rectangleOrigin - observer);
        return retval;
    }

    scalar_type solidAngleOfRectangle(NBL_CONST_REF_ARG(vector<scalar_type, 2>) rectangleExtents)
    {
        const vector4_type denorm_n_z = vector4_type(-r0.y, r0.x + rectangleExtents.x, r0.y + rectangleExtents.y, -r0.x);
        const vector4_type n_z = denorm_n_z / nbl::hlsl::sqrt((vector4_type)(r0.z * r0.z) + denorm_n_z * denorm_n_z);
        const vector4_type cosGamma = vec4(
            -n_z[0] * n_z[1],
            -n_z[1] * n_z[2],
            -n_z[2] * n_z[3],
            -n_z[3] * n_z[0]
        );
        return math::getSumofArccosABCD(cosGamma[0], cosGamma[1], cosGamma[2], cosGamma[3]) - 2 * numbers::pi<float>;
    }

    vector3_type r0;
};

}
}
}

#endif
