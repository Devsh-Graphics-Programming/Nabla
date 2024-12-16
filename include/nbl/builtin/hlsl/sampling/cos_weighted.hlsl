// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_COS_WEIGHTED_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_COS_WEIGHTED_INCLUDED_

#include "nbl/builtin/hlsl/concepts.hlsl"
#include "nbl/builtin/hlsl/sampling/concentric_mapping.glsl"

namespace nbl
{
namespace hlsl
{

namespace impl
{
template<typename T>
struct ProjectedHemisphere
{
    using vector_t2 = vector<T, 2>;
    using vector_t3 = vector<T, 3>;
    
    static vector_t3 generate(vector_t2 _sample)
    {
        vector_t2 p = concentricMapping<T>(_sample * 0.99999 + 0.000005);
        T z = sqrt<T>(max(0.0, 1.0 - p.x * p.x - p.y * p.y));
        return vector_t3(p.x, p.y, z);
    }

    static T pdf(T L_z)
    {
        return L_z * numbers::inv_pi<float>;
    }

    static T quotient_and_pdf(out T pdf, T L)
    {
        pdf = pdf(L);
        return 1.0;
    }
};
}

// probably could've split into separate structs and ProjectedSphere inherit ProjectedHemisphere
template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
vector<T, 3> projected_hemisphere_generate(vector<T, 2> _sample)
{
    return impl::ProjectedHemisphere<T>::generate(_sample);
}

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
T projected_hemisphere_pdf(T L_z)
{
    return impl::ProjectedHemisphere<T>::pdf(L_z);
}

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
T projected_hemisphere_quotient_and_pdf(out T pdf, T L)
{
    return impl::ProjectedHemisphere<T>::quotient_and_pdf(pdf, L);
}

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
T projected_hemisphere_quotient_and_pdf(out T pdf, vector<T,3> L)
{
    return impl::ProjectedHemisphere<T>::quotient_and_pdf(pdf, L.z);
}


template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
vector<T, 3> projected_sphere_generate(vector<T, 3> _sample)
{
    vector<T, 3> retval = impl::ProjectedHemisphere<T>::generate(_sample.xy);
    const bool chooseLower = _sample.z > 0.5;
    retval.z = chooseLower ? (-retval.z) : retval.z;
    if (chooseLower)
        _sample.z -= 0.5f;
    _sample.z *= 2.f;
    return retval;
}

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
T projected_sphere_pdf(T L_z)
{
    return 0.5 * impl::ProjectedHemisphere<T>::pdf(L_z);
}

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
T projected_sphere_quotient_and_pdf(out T pdf, T L)
{
    T retval = impl::ProjectedHemisphere<T>::quotient_and_pdf(pdf, L);
    pdf *= 0.5;
    return retval;
}

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
T projected_sphere_quotient_and_pdf(out T pdf, vector<T,3> L)
{
    T retval = impl::ProjectedHemisphere<T>::quotient_and_pdf(pdf, L.z);
    pdf *= 0.5;
    return retval;
}

}
}

#endif
