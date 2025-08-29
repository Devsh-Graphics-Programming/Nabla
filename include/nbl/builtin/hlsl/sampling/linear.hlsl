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

    static Linear<T> create(NBL_CONST_REF_ARG(vector2_type) linearCoeffs)
    {
        Linear<T> retval;
        retval.linearCoeffs = linearCoeffs;
        return retval;
    }

    scalar_type generate(scalar_type u)
    {
        const scalar_type rcpDiff = 1.0 / (linearCoeffs[0] - linearCoeffs[1]);
        const vector2_type squaredCoeffs = linearCoeffs * linearCoeffs;
        return nbl::hlsl::abs(rcpDiff) < numeric_limits<scalar_type>::max ? (linearCoeffs[0] - nbl::hlsl::sqrt(nbl::hlsl::mix(squaredCoeffs[0], squaredCoeffs[1], u))) * rcpDiff : u;
    }

    vector2_type linearCoeffs;
};

}
}
}

#endif
