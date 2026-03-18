// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_LINEAR_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_LINEAR_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/limits.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template<typename T>
struct Linear
{
    using scalar_type = T;
    using vector2_type = vector<T, 2>;

    // BijectiveSampler concept types
    using domain_type = scalar_type;
    using codomain_type = scalar_type;
    using density_type = scalar_type;
    using weight_type = density_type;

    struct cache_type
    {
        density_type pdf;
    };

    static Linear<T> create(const vector2_type linearCoeffs)   // start and end importance values (start, end), assumed to be at x=0 and x=1
    {
        Linear<T> retval;
        retval.linearCoeffStart = linearCoeffs[0];
        retval.linearCoeffDiff = linearCoeffs[1] - linearCoeffs[0];
        retval.rcpCoeffSum = scalar_type(1.0) / (linearCoeffs[0] + linearCoeffs[1]);
        retval.rcpDiff = -scalar_type(1.0) / retval.linearCoeffDiff;
        vector2_type squaredCoeffs = linearCoeffs * linearCoeffs;
        retval.squaredCoeffStart = squaredCoeffs[0];
        retval.squaredCoeffDiff = squaredCoeffs[1] - squaredCoeffs[0];
        return retval;
    }

    density_type __pdf(const codomain_type x)
    {
        if (x < scalar_type(0.0) || x > scalar_type(1.0))
            return scalar_type(0.0);
        return scalar_type(2.0) * (linearCoeffStart + x * linearCoeffDiff) * rcpCoeffSum;
    }

    codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache)
    {
        const codomain_type x = hlsl::mix(u, (linearCoeffStart - sqrt(squaredCoeffStart + u * squaredCoeffDiff)) * rcpDiff, abs(rcpDiff) < hlsl::numeric_limits<scalar_type>::max);
        cache.pdf = __pdf(x);
        return x;
    }

    domain_type generateInverse(const codomain_type x)
    {
        return x * (scalar_type(2.0) * linearCoeffStart + linearCoeffDiff * x) * rcpCoeffSum;
    }

    density_type forwardPdf(const cache_type cache)
    {
        return cache.pdf;
    }

    weight_type forwardWeight(const cache_type cache)
    {
        return forwardPdf(cache);
    }

    density_type backwardPdf(const codomain_type x)
    {
        return __pdf(x);
    }

    weight_type backwardWeight(const codomain_type x)
    {
        return backwardPdf(x);
    }

    scalar_type linearCoeffStart;
    scalar_type linearCoeffDiff;
    scalar_type rcpCoeffSum;
    scalar_type rcpDiff;
    scalar_type squaredCoeffStart;
    scalar_type squaredCoeffDiff;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
