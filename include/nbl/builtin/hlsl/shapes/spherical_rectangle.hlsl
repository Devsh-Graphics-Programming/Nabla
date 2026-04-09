// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SHAPES_SPHERICAL_RECTANGLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_SPHERICAL_RECTANGLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/numbers.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/math/angle_adding.hlsl>
#include <nbl/builtin/hlsl/math/fast_acos.hlsl>

namespace nbl
{
namespace hlsl
{
namespace shapes
{

// What are we likely to do with a Spherical Rectangle?
// 1) Initialize it multiple times from different observers
// 2) Sample it repeatedly

// How are we likely to get a spherical rect?
// 1) from OBB matrix (with a model space z-axis scale thats irrelevant - but should be forced to 1.f to not mess with distance)
// 2) in a compressed form

// So, to bring multiple world-space observers into Spherical Rectangle's own space, we need the basis matrix.
// The matrix should be a matrix<scalar_type,3,4> where the last column is the translation, a 3x3 matrix with a pre-transform translation (worldSpace rectangle origin to be subtracted).

// You can compute it from an OBB matrix (as given by/to imguizmo to position a [0,1]^2 rectangle mesh where Z+ is the front face.

/*
matrix<scalar_type,3,3> check = mul(modelSpace,tranpose(modelSpace));
// orthogonality (don't need to check the other 3 lower half numbers, cause MM^T is symmetric)
assert(check[0][1]==0.f);
assert(check[0][2]==0.f);
assert(check[1][2]==0.f);
// the scales are squared
const vector2_type scalesSq = vector2_type(check[0][0],check[1][1]);
const vector2_type scalesRcp = rsqrt(scalesSq);
// only rotation, scale needs to be thrown away
basis = tranpose(modelSpace);
// right now `mul(basis,fromObserver)` will apply extent scales on the dot product
// need to remove that
basis[0] *= scalesRcp[0];
basis[1] *= scalesRcp[1];
// but also back it up so we know the size of the original rectangle
extents = promote(vector2_type>(1.f)/scalesRcp;
if (dontAssertZScaleIsOne)
   basis[2] *= rsqrt(check[2][2]);
else
{
    assert(check[2][2]==1.f);
}
*/

// Now, can apply translation:
// 1) post-rotation so a it automatically gets added during a affine pseudo-mul of a 3x4, so pseudo_mul(basis,observer)
// 2) pre-rotation so you keep a worldspace rectangle origin and subtract it before, e.g. mul(basis,worldSpaceOrigin-observer) - this one is possibly better due to next point

// So we need to store:
// 1) first two COLUMNS of the original OBB matrix (rows of 3x3 basis matrix with the scale still in there), thats kinda your right and up vectors
// 2) pre-rotation translation / the world-space translation of the rectangle
// Theoretically you could get away with not storing one of the up vector components but its not always the same component you can reconstruct (plane orthogonal to up isn't always the XY plane).
// Could compress up vector as a rotation of the default vector orthogonal to right as given by the frisvad-basis function around the right vector plus a scale
// but that becomes a very expensive decompression step involving a quaternion with uniform scale.

template<typename Scalar>
struct CompressedSphericalRectangle
{
    using vector3_type = vector<Scalar, 3>;

    vector3_type origin;
    vector3_type right;
    vector3_type up;
};

template<typename Scalar>
struct SphericalRectangle
{
    using scalar_type = Scalar;
    using vector2_type = vector<Scalar, 2>;
    using vector3_type = vector<Scalar, 3>;
    using vector4_type = vector<Scalar, 4>;
    using matrix3x3_type = matrix<Scalar, 3, 3>;

    struct solid_angle_type
    {
        scalar_type value;
        vector3_type r0;
        vector4_type n_z;
        vector4_type cosGamma;
    };

    static SphericalRectangle<Scalar> create(NBL_CONST_REF_ARG(CompressedSphericalRectangle<Scalar>) compressed)
    {
        SphericalRectangle<scalar_type> retval;
        retval.origin = compressed.origin;
        retval.extents = vector2_type(hlsl::length(compressed.right), hlsl::length(compressed.up));
        retval.basis[0] = compressed.right / retval.extents[0];
        retval.basis[1] = compressed.up / retval.extents[1];
        assert(hlsl::abs(hlsl::dot(retval.basis[0], retval.basis[1])) < scalar_type(1e-5));
        retval.basis[2] = hlsl::normalize(hlsl::cross(retval.basis[0], retval.basis[1]));
        return retval;
    }

    solid_angle_type solidAngle(const vector3_type observer) NBL_CONST_MEMBER_FUNC
    {
        solid_angle_type result;
        result.r0 = hlsl::mul(basis, origin - observer);

        const vector4_type denorm_n_z = vector4_type(-result.r0.y, result.r0.x + extents.x, result.r0.y + extents.y, -result.r0.x);
        result.n_z = denorm_n_z / nbl::hlsl::sqrt(hlsl::promote<vector4_type>(result.r0.z * result.r0.z) + denorm_n_z * denorm_n_z);
        result.cosGamma = vector4_type(
            -result.n_z[0] * result.n_z[1],
            -result.n_z[1] * result.n_z[2],
            -result.n_z[2] * result.n_z[3],
            -result.n_z[3] * result.n_z[0]
        );
        math::sincos_accumulator<scalar_type> angle_adder = math::sincos_accumulator<scalar_type>::create(result.cosGamma[0]);
        angle_adder.addCosine(result.cosGamma[1]);
        angle_adder.addCosine(result.cosGamma[2]);
        angle_adder.addCosine(result.cosGamma[3]);
        result.value = angle_adder.getSumOfArccos() - scalar_type(2.0) * numbers::pi<scalar_type>;
        return result;
    }

    vector3_type origin;
    vector2_type extents;
    matrix3x3_type basis;
};

}
}
}

#endif
