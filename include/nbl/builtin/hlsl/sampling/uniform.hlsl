// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_UNIFORM_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_UNIFORM_INCLUDED_

#include "nbl/builtin/hlsl/concepts.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/tgmath.hlsl"

namespace nbl
{
namespace hlsl
{

namespace impl
{
template<typename T>
struct UniformHemisphere
{
    using vector_t2 = vector<T, 2>;
    using vector_t3 = vector<T, 3>;

    static vector_t3 generate(vector_t2 _sample)
    {
        T z = _sample.x;
        T r = sqrt<T>(max<T>(0.0, 1.0 - z * z));
        T phi = 2.0 * numbers::pi<T> * _sample.y;
        return vector_t3(r * cos<T>(phi), r * sin<T>(phi), z);
    }

    static T pdf()
    {
        return 1.0 / (2.0 * numbers::pi<float>);
    }
};

template<typename T>
struct UniformSphere
{
    using vector_t2 = vector<T, 2>;
    using vector_t3 = vector<T, 3>;

    static vector_t3 generate(vector_t2 _sample)
    {
        T z = 1 - 2 * _sample.x;
        T r = sqrt<T>(max<T>(0.0, 1.0 - z * z));
        T phi = 2.0 * numbers::pi<T> * _sample.y;
        return vector_t3(r * cos<T>(phi), r * sin<T>(phi), z);
    }

    static T pdf()
    {
        return 1.0 / (4.0 * numbers::pi<float>);
    }
};
}

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
vector<T, 3> uniform_hemisphere_generate(vector<T, 2> _sample)
{
    return impl::UniformHemisphere<T>::generate(_sample);
}

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
T uniform_hemisphere_pdf()
{
    return impl::UniformHemisphere<T>::pdf();
}

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
vector<T, 3> uniform_sphere_generate(vector<T, 2> _sample)
{
    return impl::UniformSphere<T>::generate(_sample);
}

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
T uniform_sphere_pdf()
{
    return impl::UniformSphere<T>::pdf();
}

}
}

#endif
