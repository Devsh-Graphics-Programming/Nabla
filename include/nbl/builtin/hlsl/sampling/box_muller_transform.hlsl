// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_BOX_MULLER_TRANSFORM_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_BOX_MULLER_TRANSFORM_INCLUDED_

#include "nbl/builtin/hlsl/math/functions.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/sampling/value_and_pdf.hlsl"

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template<typename T NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeScalar<T>)
struct BoxMullerTransform
{
    using scalar_type = T;
    using vector2_type = vector<T, 2>;

    // BackwardTractableSampler concept types
    using domain_type = vector2_type;
    using codomain_type = vector2_type;
    using density_type = scalar_type;
    using weight_type = density_type;

    struct cache_type
    {
        vector2_type direction; // (cosPhi, sinPhi)
        scalar_type u_x;
    };

    static BoxMullerTransform<T> create(const scalar_type _stddev)
    {
        BoxMullerTransform<T> retval;
        retval.stddev = _stddev;
        retval.halfRcpStddev2 = scalar_type(0.5) / (_stddev * _stddev);
        return retval;
    }

    codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
    {
        scalar_type sinPhi, cosPhi;
        math::sincos<scalar_type>(scalar_type(2.0) * numbers::pi<scalar_type> * u.y - numbers::pi<scalar_type>, sinPhi, cosPhi);
        cache.direction = vector2_type(cosPhi, sinPhi);
        cache.u_x = u.x;
        return cache.direction * nbl::hlsl::sqrt(scalar_type(-2.0) * nbl::hlsl::log(u.x)) * stddev;
    }

    density_type forwardPdf(const cache_type cache) NBL_CONST_MEMBER_FUNC
    {
        return halfRcpStddev2 * numbers::inv_pi<scalar_type> * cache.u_x;
    }

    vector2_type separateForwardPdf(const cache_type cache) NBL_CONST_MEMBER_FUNC
    {
        const scalar_type normalization = nbl::hlsl::sqrt(halfRcpStddev2 * numbers::inv_pi<scalar_type>);
        const vector2_type dir2 = cache.direction * cache.direction;
        return normalization * vector2_type(
            nbl::hlsl::pow(cache.u_x, dir2.x),
            nbl::hlsl::pow(cache.u_x, dir2.y)
        );
    }

    weight_type forwardWeight(const cache_type cache) NBL_CONST_MEMBER_FUNC
    {
        return forwardPdf(cache);
    }

    density_type backwardPdf(const codomain_type outPos) NBL_CONST_MEMBER_FUNC
    {
        const scalar_type normalization = halfRcpStddev2 * numbers::inv_pi<scalar_type>;
        return normalization * nbl::hlsl::exp(-halfRcpStddev2 * nbl::hlsl::dot(outPos, outPos));
    }

    vector2_type separateBackwardPdf(const codomain_type outPos) NBL_CONST_MEMBER_FUNC
    {
        const scalar_type normalization = nbl::hlsl::sqrt(halfRcpStddev2 * numbers::inv_pi<scalar_type>);
        const vector2_type outPos2 = outPos * outPos;
        return vector2_type(
            normalization * nbl::hlsl::exp(-halfRcpStddev2 * outPos2.x),
            normalization * nbl::hlsl::exp(-halfRcpStddev2 * outPos2.y)
        );
    }

    weight_type backwardWeight(const codomain_type outPos) NBL_CONST_MEMBER_FUNC
    {
        return backwardPdf(outPos);
    }

    T stddev;
    T halfRcpStddev2;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
