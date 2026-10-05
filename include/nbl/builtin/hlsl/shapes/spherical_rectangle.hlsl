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

    // TODO: take an observer already, this way we can precompute and store the `r0`, `denorm_n_z` and `rcpLen_denorm_n_z`
    // we need all of the above for solid angle and projected solid angle computation
    static SphericalRectangle<Scalar> create(NBL_CONST_REF_ARG(CompressedSphericalRectangle<Scalar>) compressed)
    {
        SphericalRectangle<scalar_type> retval;
        retval.origin = compressed.origin;
        retval.extents = vector2_type(hlsl::length(compressed.right), hlsl::length(compressed.up));
        retval.basis[0] = compressed.right / retval.extents[0];
        retval.basis[1] = compressed.up / retval.extents[1];
        assert(hlsl::abs(hlsl::dot(retval.basis[0], retval.basis[1])) < scalar_type(1e-5));
        // don't normalize, the `right` and `up` vectors are orthogonal and the two bases are normalized already!
        retval.basis[2] = hlsl::cross(retval.basis[0], retval.basis[1]);
        return retval;
    }

    solid_angle_type solidAngle(const vector3_type observer) NBL_CONST_MEMBER_FUNC
    {
        solid_angle_type result;
        result.r0 = hlsl::mul(basis, origin - observer);

        const vector4_type denorm_n_z = vector4_type(-result.r0.y, result.r0.x + extents.x, result.r0.y + extents.y, -result.r0.x);
        const vector4_type rcpLen_denorm_n_z = hlsl::rsqrt<vector4_type>(hlsl::promote<vector4_type>(result.r0.z * result.r0.z) + denorm_n_z * denorm_n_z);
        result.n_z = denorm_n_z * rcpLen_denorm_n_z;
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

    // Kelvin-Stokes theorem: signed projected solid angle = integral_{rect} (n . omega) d_omega
    // TODO: don't take the observer, observer should be taken at creation
    scalar_type projectedSolidAngle(const vector3_type observer, const vector3_type receiverNormal) NBL_CONST_MEMBER_FUNC
    {
        return projectedSolidAngleFromLocal(hlsl::mul(basis, origin - observer), hlsl::mul(basis, receiverNormal));
    }

    // TODO: only take a `localN`
    scalar_type projectedSolidAngleFromLocal(const vector3_type r0, const vector3_type n) NBL_CONST_MEMBER_FUNC
    {
        // FUN FACT: `n_z` already holds Z coordinate the NORMALIZED `awayFromEdgePlane`, the non-zero coordinate absolute value is equal to `r0.z * rcpLen_denorm_n_z`
// TODO: skip all this code until  just call `acos` on the `unnormDots`
        const scalar_type x0 = r0.x, y0 = r0.y, z = r0.z;
        const scalar_type x1 = x0 + extents.x;
        const scalar_type y1 = y0 + extents.y;
        const scalar_type ex = extents.x, ey = extents.y;
        const scalar_type zSq = z * z;

        // Unnormalized cross products of adjacent corners (each has one zero component):
        //   cross(r0,r1) = ex*(0, z, -y0),  cross(r1,r2) = ey*(-z, 0, x1)
        //   cross(r2,r3) = ex*(0, -z, y1),  cross(r3,r0) = ey*(z, 0, -x0)
        // |cross|^2: the ex/ey factors cancel in externalProducts (dot/|cross|)
        const vector4_type crossLenSq = vector4_type(
            zSq + y0 * y0,
            zSq + x1 * x1,
            zSq + y1 * y1,
            zSq + x0 * x0
        ); // TODO: this is already computed as `rcpLen_denorm_n_z`

// TODO: this can be computed from `denorm_n_z`, `z` and `rcpLen_denorm_n_z` instead
        // dot(cross(ri,rj), n) / |cross(ri,rj)| the ex/ey scale factors cancel
        const vector4_type crossDotN = vector4_type(
            z * n.y - y0 * n.z,
            -z * n.x + x1 * n.z,
            -z * n.y + y1 * n.z,
            z * n.x - x0 * n.z
        );
        // The ABS makes the computation correct for abs(cos(theta)) (BSDF projected solid angle).
        const vector4_type externalProducts = crossDotN * hlsl::rsqrt<vector4_type>(crossLenSq);

// TODO: isn't `rcpLen_denorm_n_z` related to the sin^-1() of arclengths ? Wouldn't `ACOS_CSC` apply instead  of `acos*rsqrt(1-cos^2)`
// wouldn't then `hlsl::promote<vector4_type>(result.r0.z * result.r0.z) + denorm_n_z * denorm_n_z` be the sin^2 ?
// it would probably have to be a different, here's the `acos(sqrt(1-x*x))/x` curve fit again revealed to me in a dream
//  https://www.desmos.com/calculator/sbdrulot5a = exp2(-1.6*sqrt(1-x*x))*A+B
        // cos(arc length) between adjacent corners: dot(ri,rj) / (|ri|*|rj|)
        const vector4_type lenSq = vector4_type(
            x0 * x0 + y0 * y0,
            x1 * x1 + y0 * y0,
            x1 * x1 + y1 * y1,
            x0 * x0 + y1 * y1
        ) + hlsl::promote<vector4_type>(zSq);
        const vector4_type rcpLen = hlsl::rsqrt<vector4_type>(lenSq);

        const vector4_type unnormDots = vector4_type(
            x0 * x1 + y0 * y0 + zSq,
            x1 * x1 + y0 * y1 + zSq,
            x0 * x1 + y1 * y1 + zSq,
            x0 * x0 + y0 * y1 + zSq
        );
        // rcpLen[i]*rcpLen[j] for adjacent pairs: (0,1), (1,2), (2,3), (3,0)
        const vector4_type cos_sides = unnormDots * rcpLen * rcpLen.yzwx;

        // TODO: there's the same opportunity for optimization of this as the Spherical Triangle
        // https://www.linkedin.com/posts/matt-kielan-9b054a165_untitled-graph-activity-7442910005671923712-jHz6?utm_source=share&utm_medium=member_desktop&rcm=ACoAACdp2RQBqq2bJfC2zxpsme-vRv2zh9oP-8E
        const vector4_type pyramidAngles = hlsl::acos<vector4_type>(cos_sides);

        return hlsl::abs(hlsl::dot(pyramidAngles, externalProducts)) * scalar_type(0.5);
    }

    vector3_type origin;
    vector2_type extents;
    matrix3x3_type basis;
};

}
}
}

#endif
