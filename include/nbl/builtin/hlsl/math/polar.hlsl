// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MATH_POLAR_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_POLAR_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"

namespace nbl
{
namespace hlsl
{
namespace math
{

template<typename T = float32_t>
struct Polar
{
    using scalar_type = T;
    using vector2_type = vector<T, 2>;
    using vector3_type = vector<T, 3>;

    // input must be normalized
    static Polar<T> createFromCartesian(const vector3_type dir)
    {
        Polar<T> retval;
        retval.theta = hlsl::acos<scalar_type>(dir.z);
        retval.phi = hlsl::atan2<scalar_type>(dir.y, dir.x);
        return retval;
    }

    static vector3_type ToCartesian(const scalar_type theta, const scalar_type phi)
    {
        return vector<T, 3>(
            hlsl::cos<scalar_type>(phi) * hlsl::cos<scalar_type>(theta),
            hlsl::sin<scalar_type>(phi) * hlsl::cos<scalar_type>(theta),
            hlsl::sin<scalar_type>(theta)
        );
    }

    vector3_type getCartesian()
    {
        return ToCartesian(theta, phi);
    }

    scalar_type theta;  //! polar angle
    scalar_type phi;    //! azimuthal angle
};

}
}
}

#endif
