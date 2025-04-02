// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_COS_WEIGHTED_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_COS_WEIGHTED_INCLUDED_

#include "nbl/builtin/hlsl/concepts.hlsl"
#include "nbl/builtin/hlsl/sampling/concentric_mapping.hlsl"
#include "nbl/builtin/hlsl/bxdf/common.hlsl"

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
struct ProjectedHemisphere
{
    using vector_t2 = vector<T, 2>;
    using vector_t3 = vector<T, 3>;
    
    static vector_t3 generate(vector_t2 _sample)
    {
        vector_t2 p = concentricMapping<T>(_sample * T(0.99999) + T(0.000005));
        T z = hlsl::sqrt<T>(hlsl::max<T>(T(0.0), T(1.0) - p.x * p.x - p.y * p.y));
        return vector_t3(p.x, p.y, z);
    }

    static T pdf(T L_z)
    {
        return L_z * numbers::inv_pi<float>;
    }

    template<typename SpectralBins>
    static bxdf::quotient_and_pdf<SpectralBins, T> quotient_and_pdf(T L)
    {
        return bxdf::quotient_and_pdf<SpectralBins, T>::create(hlsl::promote<SpectralBins>(1.0), pdf(L));
    }

    template<typename SpectralBins>
    static bxdf::quotient_and_pdf<SpectralBins, T> quotient_and_pdf(vector_t3 L)
    {
        return quotient_and_pdf<SpectralBins>(L.z);
    }
};

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
struct ProjectedSphere
{
    using vector_t2 = vector<T, 2>;
    using vector_t3 = vector<T, 3>;
    using hemisphere_t = ProjectedHemisphere<T>;

    static vector_t3 generate(vector_t2 _sample)
    {
        vector_t3 retval = hemisphere_t::generate(_sample.xy);
        const bool chooseLower = _sample.z > T(0.5);
        retval.z = chooseLower ? (-retval.z) : retval.z;
        if (chooseLower)
            _sample.z -= T(0.5);
        _sample.z *= T(2.0);
        return retval;
    }

    static T pdf(T L_z)
    {
        return T(0.5) * hemisphere_t::pdf(L_z);
    }

    template<typename SpectralBins>
    static bxdf::quotient_and_pdf<SpectralBins, T> quotient_and_pdf(T L)
    {
        return bxdf::quotient_and_pdf<SpectralBins, T>::create(hlsl::promote<SpectralBins>(1.0), pdf(L));
    }

    template<typename SpectralBins>
    static bxdf::quotient_and_pdf<SpectralBins, T> quotient_and_pdf(vector_t3 L)
    {
        return quotient_and_pdf<SpectralBins>(L.z);
    }
}

}
}
}

#endif
