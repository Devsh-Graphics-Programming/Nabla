// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_UNIFORM_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_UNIFORM_INCLUDED_

#include "nbl/builtin/hlsl/concepts.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/tgmath.hlsl"
#include "nbl/builtin/hlsl/sampling/quotient_and_pdf.hlsl"

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
struct UniformHemisphere
{
    using vector_t2 = vector<T, 2>;
    using vector_t3 = vector<T, 3>;

    static vector_t3 generate(vector_t2 _sample)
    {
        T z = _sample.x;
        T r = hlsl::sqrt<T>(hlsl::max<T>(T(0.0), T(1.0) - z * z));
        T phi = T(2.0) * numbers::pi<T> * _sample.y;
        return vector_t3(r * hlsl::cos<T>(phi), r * hlsl::sin<T>(phi), z);
    }

    static T pdf()
    {
        return T(1.0) / (T(2.0) * numbers::pi<T>);
    }

    template<typename U=T NBL_FUNC_REQUIRES(is_scalar_v<U>)
    static quotient_and_pdf<U, T> quotient_and_pdf()
    {
        return quotient_and_pdf<U, T>::create(U(1.0), pdf());
    }
};

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
struct UniformSphere
{
    using vector_t2 = vector<T, 2>;
    using vector_t3 = vector<T, 3>;

    static vector_t3 generate(vector_t2 _sample)
    {
        T z = T(1.0) - T(2.0) * _sample.x;
        T r = hlsl::sqrt<T>(hlsl::max<T>(T(0.0), T(1.0) - z * z));
        T phi = T(2.0) * numbers::pi<T> * _sample.y;
        return vector_t3(r * hlsl::cos<T>(phi), r * hlsl::sin<T>(phi), z);
    }

    static T pdf()
    {
        return T(1.0) / (T(4.0) * numbers::pi<T>);
    }

    template<typename U=T NBL_FUNC_REQUIRES(is_scalar_v<U>)
    static quotient_and_pdf<U, T> quotient_and_pdf()
    {
        return quotient_and_pdf<U, T>::create(U(1.0), pdf());
    }
};
}

}
}

#endif
