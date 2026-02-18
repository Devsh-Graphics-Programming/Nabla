// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_COS_WEIGHTED_SPHERES_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_COS_WEIGHTED_SPHERES_INCLUDED_

#include "nbl/builtin/hlsl/concepts.hlsl"
#include "nbl/builtin/hlsl/sampling/concentric_mapping.hlsl"
#include "nbl/builtin/hlsl/sampling/quotient_and_pdf.hlsl"

namespace nbl
{
    namespace hlsl
    {
        namespace sampling
        {

            template <typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>) struct ProjectedHemisphere
            {
                using vector_t2 = vector<T, 2>;
                using vector_t3 = vector<T, 3>;

                // BijectiveSampler concept types
                using scalar_type = T;
                using domain_type = vector_t2;
                using codomain_type = vector_t3;
                using density_type = T;
                using sample_type = codomain_and_rcpPdf<codomain_type, density_type>;
                using inverse_sample_type = domain_and_rcpPdf<domain_type, density_type>;

                static vector_t3 generate(const vector_t2 _sample)
                {
                    vector_t2 p = concentricMapping<T>(_sample * T(0.99999) + T(0.000005));
                    T z = hlsl::sqrt<T>(hlsl::max<T>(T(0.0), T(1.0) - p.x * p.x - p.y * p.y));
                    return vector_t3(p.x, p.y, z);
                }

                static T pdf(const T L_z)
                {
                    return L_z * numbers::inv_pi<float>;
                }

                template <typename U = vector<T, 1>>
                static ::nbl::hlsl::sampling::quotient_and_pdf<U, T> quotient_and_pdf(const T L)
                {
                    return ::nbl::hlsl::sampling::quotient_and_pdf<U, T>::create(hlsl::promote<U>(1.0), pdf(L));
                }

                template <typename U = vector<T, 1>>
                static ::nbl::hlsl::sampling::quotient_and_pdf<U, T> quotient_and_pdf(const vector_t3 L)
                {
                    return ::nbl::hlsl::sampling::quotient_and_pdf<U, T>::create(hlsl::promote<U>(1.0), pdf(L.z));
                }
            };

            template <typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>) struct ProjectedSphere
            {
                using vector_t2 = vector<T, 2>;
                using vector_t3 = vector<T, 3>;
                using hemisphere_t = ProjectedHemisphere<T>;

                // BijectiveSampler concept types
                using scalar_type = T;
                using domain_type = vector_t3;
                using codomain_type = vector_t3;
                using density_type = T;
                using sample_type = codomain_and_rcpPdf<codomain_type, density_type>;
                using inverse_sample_type = domain_and_rcpPdf<domain_type, density_type>;

                static vector_t3 generate(NBL_REF_ARG(vector_t3) _sample)
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

                template <typename U = vector<T, 1>>
                static ::nbl::hlsl::sampling::quotient_and_pdf<U, T> quotient_and_pdf(T L)
                {
                    return ::nbl::hlsl::sampling::quotient_and_pdf<U, T>::create(hlsl::promote<U>(1.0), pdf(L));
                }

                template <typename U = vector<T, 1>>
                static ::nbl::hlsl::sampling::quotient_and_pdf<U, T> quotient_and_pdf(const vector_t3 L)
                {
                    return ::nbl::hlsl::sampling::quotient_and_pdf<U, T>::create(hlsl::promote<U>(1.0), pdf(L.z));
                }
            };

        }
    }
}

#endif
