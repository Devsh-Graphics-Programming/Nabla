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

    // InvertibleSampler concept types
    using domain_type = vector2_type;
    using codomain_type = vector2_type;
    using density_type = scalar_type;
    using weight_type = density_type;

    struct cache_type
    {
        density_type pdf;
    };

    codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache)
    {
        scalar_type sinPhi, cosPhi;
        math::sincos<scalar_type>(scalar_type(2.0) * numbers::pi<scalar_type> * u.y - numbers::pi<scalar_type>, sinPhi, cosPhi);
        const codomain_type outPos = vector2_type(cosPhi, sinPhi) * nbl::hlsl::sqrt(scalar_type(-2.0) * nbl::hlsl::log(u.x)) * stddev;
        cache.pdf = backwardPdf(outPos);
        return outPos;
    }

    density_type forwardPdf(const cache_type cache)
    {
        return cache.pdf;
    }

    vector2_type separateForwardPdf(const cache_type cache, const codomain_type outPos)
    {
        return separateBackwardPdf(outPos);
    }

    weight_type forwardWeight(const cache_type cache)
    {
        return forwardPdf(cache);
    }

    density_type backwardPdf(const codomain_type outPos)
    {
        const vector2_type marginals = separateBackwardPdf(outPos);
        return marginals.x * marginals.y;
    }

    vector2_type separateBackwardPdf(const codomain_type outPos)
    {
        const scalar_type stddev2 = stddev * stddev;
        const scalar_type normalization = scalar_type(1.0) / (stddev * nbl::hlsl::sqrt(scalar_type(2.0) * numbers::pi<scalar_type>));
        const vector2_type outPos2 = outPos * outPos;
        return vector2_type(
            normalization * nbl::hlsl::exp(scalar_type(-0.5) * outPos2.x / stddev2),
            normalization * nbl::hlsl::exp(scalar_type(-0.5) * outPos2.y / stddev2)
        );
    }

    weight_type backwardWeight(const codomain_type outPos)
    {
        return backwardPdf(outPos);
    }

    T stddev;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
