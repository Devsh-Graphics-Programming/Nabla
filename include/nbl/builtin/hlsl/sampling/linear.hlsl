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

    static Linear<T> create(const vector2_type linearCoeffs)   // start and end importance values (start, end), assumed to be at x=0 and x=1
    {
        Linear<T> retval;
        scalar_type rcpDiff = 1.0 / (linearCoeffs[0] - linearCoeffs[1]);
        retval.linearCoeffStartOverDiff = linearCoeffs[0] * rcpDiff;
        vector2_type squaredCoeffs = linearCoeffs * linearCoeffs;
        scalar_type squaredRcpDiff = rcpDiff * rcpDiff;
        retval.squaredCoeffStartOverDiff = squaredCoeffs[0] * squaredRcpDiff;
        retval.squaredCoeffDiffOverDiff = (squaredCoeffs[1] - squaredCoeffs[0]) * squaredRcpDiff;
        return retval;
    }

    scalar_type generate(const scalar_type u)
    {
        return hlsl::mix(u, (linearCoeffStartOverDiff - hlsl::sqrt(squaredCoeffStartOverDiff + u * squaredCoeffDiffOverDiff)), hlsl::abs(linearCoeffStartOverDiff) < numeric_limits<scalar_type>::max);
    }

    // TODO: add forwardPdf and backwardPdf methods, forward computes from u and backwards from the result of generate

    scalar_type linearCoeffStartOverDiff;  
    scalar_type squaredCoeffStartOverDiff;
    scalar_type squaredCoeffDiffOverDiff;
};

}
}
}

#endif
