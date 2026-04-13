// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_SPHERICAL_RECTANGLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_SPHERICAL_RECTANGLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/math/fast_acos.hlsl>
#include <nbl/builtin/hlsl/ieee754.hlsl>
#include <nbl/builtin/hlsl/shapes/spherical_rectangle.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template<typename T>
struct SphericalRectangle
{
    using scalar_type = T;
    using vector2_type = vector<T, 2>;
    using vector3_type = vector<T, 3>;
    using vector4_type = vector<T, 4>;

    // BackwardTractableSampler concept types
    using domain_type = vector2_type;
    using codomain_type = vector3_type;
    using density_type = scalar_type;
    using weight_type = density_type;

    struct cache_type {};

    NBL_CONSTEXPR_STATIC_INLINE scalar_type ClampEps = 1e-5;

    static SphericalRectangle<T> create(NBL_CONST_REF_ARG(shapes::SphericalRectangle<T>) rect, const vector3_type observer)
    {
        return create(rect.solidAngle(observer), rect.extents);
    }

    static SphericalRectangle<T> create(NBL_CONST_REF_ARG(typename shapes::SphericalRectangle<T>::solid_angle_type) sa, const vector2_type _extents)
    {
        SphericalRectangle<T> retval;
        retval.r0 = sa.r0;
        retval.extents = _extents;

        retval.solidAngle = sa.value;
        retval.b0 = sa.n_z[0];
        retval.b1 = sa.n_z[2];

        math::sincos_accumulator<scalar_type> angle_adder = math::sincos_accumulator<scalar_type>::create(sa.cosGamma[2]);
        angle_adder.addCosine(sa.cosGamma[3]);
        retval.k = scalar_type(2.0) * numbers::pi<scalar_type> - angle_adder.getSumOfArccos();

        return retval;
    }

    // shared core of generate and generateSurfaceOffset
    // returns (xu, hv, d) packed into a vector3; caller derives either 2D offset or 3D direction
    vector3_type __generate(const domain_type u) NBL_CONST_MEMBER_FUNC
    {
        // algorithm needs r0.z < 0; use -abs(r0.z) without storing the flip
        const scalar_type negAbsR0z = -hlsl::abs(r0.z);
        const scalar_type r0zSq = r0.z * r0.z;
        const vector2_type r1 = vector2_type(r0.x + extents.x, r0.y + extents.y);

        const scalar_type au = u.x * solidAngle + k;
        const scalar_type cos_au = hlsl::cos<scalar_type>(au);
        const scalar_type numerator = b1 - cos_au * b0;
        // (1-cos)*(1+cos) avoids catastrophic cancellation of 1-cos^2 when cos_au is near +/-1
        const scalar_type sin_au_sq = (scalar_type(1.0) - cos_au) * (scalar_type(1.0) + cos_au);
        const scalar_type absNegFu = hlsl::abs(numerator) * hlsl::rsqrt<scalar_type>(sin_au_sq);
        const scalar_type rcpCu_2 = hlsl::max<scalar_type>(absNegFu * absNegFu + b0 * b0, scalar_type(1.0));
        // sign(negFu) = sign(numerator) * sign(sin(au)); sin(au) < 0 iff au > PI
        const scalar_type negFuSign = hlsl::select((au > numbers::pi<scalar_type>) != (numerator < scalar_type(0.0)), scalar_type(-1.0), scalar_type(1.0));
        scalar_type xu = negAbsR0z * negFuSign * hlsl::rsqrt<scalar_type>(rcpCu_2 - scalar_type(1.0));
        xu = hlsl::clamp<scalar_type>(xu, r0.x, r1.x); // avoid Infs
        const scalar_type d_2 = xu * xu + r0zSq;
        const scalar_type d = hlsl::sqrt<scalar_type>(d_2);

        const scalar_type h0 = r0.y * hlsl::rsqrt<scalar_type>(d_2 + r0.y * r0.y);
        const scalar_type h1 = r1.y * hlsl::rsqrt<scalar_type>(d_2 + r1.y * r1.y);
        const scalar_type hv = h0 + u.y * (h1 - h0);

        return vector3_type(xu, hv, d);
    }

    // returns a normalized 3D direction in the local frame with correct r0.z sign
    codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
    {
        const vector3_type core = __generate(u);
        const scalar_type xu = core.x;
        const scalar_type hv = core.y;
        const scalar_type d = core.z;
        const scalar_type hv2 = hv * hv;
        const scalar_type cosElevation = hlsl::sqrt<scalar_type>(hlsl::max<scalar_type>(scalar_type(1.0) - hv2, scalar_type(0.0)));
        const scalar_type rcpD = scalar_type(1.0) / d;

        return vector3_type(xu * cosElevation * rcpD, hv, r0.z * cosElevation * rcpD);
    }

    // returns a 2D offset on the rectangle surface from the r0 corner
    vector2_type generateSurfaceOffset(const domain_type u, NBL_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
    {
        const vector3_type core = __generate(u);
        const scalar_type xu = core.x;
        const scalar_type hv = core.y;
        const scalar_type d = core.z;
        const scalar_type r1y = r0.y + extents.y;
        const scalar_type hv2 = hv * hv;
        const scalar_type yv = hlsl::mix(r1y, (hv * d) / hlsl::sqrt<scalar_type>(scalar_type(1.0) - hv2), hv2 < scalar_type(1.0) - ClampEps);

        return vector2_type((xu - r0.x), (yv - r0.y));
    }

    density_type forwardPdf(const domain_type u, const cache_type cache) NBL_CONST_MEMBER_FUNC
    {
        return scalar_type(1.0) / solidAngle;
    }

    weight_type forwardWeight(const domain_type u, const cache_type cache) NBL_CONST_MEMBER_FUNC
    {
        return forwardPdf(u, cache);
    }

    density_type backwardPdf(const codomain_type L) NBL_CONST_MEMBER_FUNC
    {
        return scalar_type(1.0) / solidAngle;
    }

    weight_type backwardWeight(const codomain_type L) NBL_CONST_MEMBER_FUNC
    {
        return backwardPdf(L);
    }

    scalar_type solidAngle;
    scalar_type k;
    scalar_type b0;
    scalar_type b1;
    vector3_type r0;
    vector2_type extents;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
