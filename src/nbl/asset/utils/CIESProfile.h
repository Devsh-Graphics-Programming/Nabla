// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_IES_PROFILE_H_INCLUDED__
#define __NBL_ASSET_C_IES_PROFILE_H_INCLUDED__

#include "nbl/asset/metadata/CIESProfileMetadata.h"
#include "nbl/builtin/hlsl/ies/sampler.hlsl"
#include <sstream>

namespace nbl 
{
    namespace asset 
    {
        class CIESProfile 
        {
            public:
                struct properties_t : public nbl::hlsl::ies::ProfileProperties
                {
                    hlsl::uint32_t2 optimalIESResolution; //! Optimal resolution for IES CDC texture
                };

                struct accessor_t
                {
                    using key_t = uint32_t;
                    using key_t2 = hlsl::uint32_t2;
                    using value_t = hlsl::float32_t;

                    accessor_t(const key_t2& resolution, const properties_t& props) : hAngles(resolution.x), vAngles(resolution.y), data(resolution.x * resolution.y), properties(props) {}

                    template<typename T NBL_FUNC_REQUIRES(hlsl::is_same_v<T, key_t>)
                    inline value_t vAngle(T j) const { return (value_t)vAngles[j]; }

                    template<typename T NBL_FUNC_REQUIRES(hlsl::is_same_v<T, key_t>)
                    inline value_t hAngle(T i) const { return (value_t)hAngles[i]; }

                    template<typename T NBL_FUNC_REQUIRES(hlsl::is_same_v<T, key_t2>)
                    inline value_t value(T ij) const { return (value_t)data[vAnglesCount() * ij.x + ij.y]; }

                    template<typename T NBL_FUNC_REQUIRES(hlsl::is_same_v<T, key_t2>)
                    inline void setValue(T ij, value_t val) { data[vAnglesCount() * ij.x + ij.y] = val; }

                    inline key_t vAnglesCount() { return (key_t)vAngles.size(); }
                    inline key_t hAnglesCount() { return (key_t)hAngles.size(); }
                    inline properties_t::LuminairePlanesSymmetry symmetry() { return properties.symmetry; }

                    core::vector<value_t> hAngles;          //! The angular displacement indegreesfrom straight down, a value represents spherical coordinate "theta" with physics convention. Note that if symmetry is OTHER_HALF_SYMMETRIC then real horizontal angle provided by IES data is (hAngles[index] + 90) - the reason behind it is we patch 1995 IES OTHER_HALF_SYMETRIC case to be HALF_SYMETRIC
                    core::vector<value_t> vAngles;          //! Measurements in degrees of angular displacement measured counterclockwise in a horizontal plane for Type C photometry and clockwise for Type A and B photometry, a value represents spherical coordinate "phi" with physics convention
                    core::vector<value_t> data;             //! Candela scalar values
                    properties_t properties;                //! Profile properties
                };

                using sampler_t = nbl::hlsl::ies::CandelaSampler<accessor_t>;

                CIESProfile() = default;
                ~CIESProfile() = default;
         
				inline const accessor_t& getAccessor() const { return accessor; }

                inline hlsl::float32_t getAvgEmmision(const bool fullDomain=false) const
                {
                    if (fullDomain)
                    {
                        const float cosLo = std::cos(core::radians(accessor.vAngles.front()));
                        const float cosHi = std::cos(core::radians<float>(accessor.vAngles.back()));
                        const float dsinTheta = cosLo - cosHi;
                        return accessor.properties.totalEmissionIntegral*(0.5/core::PI<float>())/dsinTheta;
                    }
                    return accessor.properties.avgEmmision;
                }

                template<class ExecutionPolicy>
                core::smart_refctd_ptr<asset::ICPUImageView> createIESTexture(ExecutionPolicy&& policy, const float flatten = 0.0, const bool fullDomainFlatten=false, uint32_t width = properties_t::CDC_DEFAULT_TEXTURE_WIDTH, uint32_t height = properties_t::CDC_DEFAULT_TEXTURE_HEIGHT) const;
                core::smart_refctd_ptr<asset::ICPUImageView> createIESTexture(const float flatten = 0.0, const bool fullDomainFlatten=false, uint32_t width = properties_t::CDC_DEFAULT_TEXTURE_WIDTH, uint32_t height = properties_t::CDC_DEFAULT_TEXTURE_HEIGHT) const;

            private:
                CIESProfile(const properties_t& props, const hlsl::uint32_t2& resolution) : accessor(resolution, props) {}
                accessor_t accessor; //! IES profile data accessor
                friend class CIESProfileParser;
        };
    }
}

#endif // __NBL_ASSET_C_IES_PROFILE_H_INCLUDED__