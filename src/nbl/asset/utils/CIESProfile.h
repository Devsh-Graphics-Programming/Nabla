// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_IES_PROFILE_H_INCLUDED__
#define __NBL_ASSET_C_IES_PROFILE_H_INCLUDED__

#include "nbl/asset/metadata/CIESProfileMetadata.h"
#include "nbl/builtin/hlsl/ies/profile.hlsl"
namespace nbl { namespace hlsl { namespace ies { struct SProceduralTexture; } } }

namespace nbl 
{
namespace asset 
{
class CIESProfile 
{
    public:
        CIESProfile() = default;
        ~CIESProfile() = default;

        struct properties_t : public nbl::hlsl::ies::ProfileProperties
        {
			using base_t = nbl::hlsl::ies::ProfileProperties;
            NBL_CONSTEXPR_STATIC_INLINE auto IES_TEXTURE_STORAGE_FORMAT = asset::EF_R16_UNORM;
            hlsl::uint32_t2 optimalIESResolution; //! Optimal resolution for IES Octahedral Candela Map texture
        };

        struct accessor_t
        {
            using angle_t = hlsl::float32_t;
            using candela_t = hlsl::float32_t;

            accessor_t() = default;
            accessor_t(const hlsl::uint32_t2& resolution, const properties_t& props) : hAngles(resolution.x), vAngles(resolution.y), data(resolution.x * resolution.y), properties(props) {}
            ~accessor_t() = default;

            inline angle_t vAngle(const uint32_t idx) const { return vAngles[idx]; }
            inline angle_t hAngle(const uint32_t idx) const { return hAngles[idx]; }
            inline uint32_t vAnglesCount() const { return static_cast<uint32_t>(vAngles.size()); }
            inline uint32_t hAnglesCount() const { return static_cast<uint32_t>(hAngles.size()); }
            inline candela_t value(hlsl::uint32_t2 ij) const { const uint32_t vCount = static_cast<uint32_t>(vAngles.size()); return data[vCount * ij.x + ij.y]; }
            inline void setValue(hlsl::uint32_t2 ij, candela_t val) { const uint32_t vCount = static_cast<uint32_t>(vAngles.size()); data[vCount * ij.x + ij.y] = val; }

			inline properties_t::base_t getProperties() const { return properties; }

            core::vector<angle_t> hAngles;          //! The angular displacement indegreesfrom straight down, a value represents spherical coordinate "theta" with physics convention. Note that if symmetry is OTHER_HALF_SYMMETRIC then real horizontal angle provided by IES data is (hAngles[index] + 90) - the reason behind it is we patch 1995 IES OTHER_HALF_SYMETRIC case to be HALF_SYMETRIC
            core::vector<angle_t> vAngles;          //! Measurements in degrees of angular displacement measured counterclockwise in a horizontal plane for Type C photometry and clockwise for Type A and B photometry, a value represents spherical coordinate "phi" with physics convention
            core::vector<candela_t> data;           //! Candela scalar values
            properties_t properties;                //! Profile properties
        };
		using texture_t = nbl::hlsl::ies::SProceduralTexture;
         
		inline const accessor_t& getAccessor() const { return accessor; }
        inline float getMaxCandelaValue() const { return accessor.properties.maxCandelaValue; }
        inline hlsl::uint32_t2 getOptimalIESResolution() const { return accessor.properties.optimalIESResolution; }
        inline float getAvgEmmision(const bool fullDomain) const { return fullDomain ? accessor.properties.fullDomainAvgEmission : accessor.properties.avgEmmision; }

        template<class ExecutionPolicy>
        core::smart_refctd_ptr<asset::ICPUImageView> createIESTexture(ExecutionPolicy&& policy, hlsl::uint32_t2 resolution) const;
        core::smart_refctd_ptr<asset::ICPUImageView> createIESTexture(hlsl::uint32_t2 resolution) const;

        template<class ExecutionPolicy>
        inline core::smart_refctd_ptr<asset::ICPUImageView> createIESTexture(ExecutionPolicy&& policy) const { const auto res = getOptimalIESResolution(); return createIESTexture(policy, res); }
        inline core::smart_refctd_ptr<asset::ICPUImageView> createIESTexture() const { const auto res = getOptimalIESResolution(); return createIESTexture(res); }

    private:
        CIESProfile(const properties_t& props, const hlsl::uint32_t2& resolution) : accessor(resolution, props) {}
        accessor_t accessor;
        friend class CIESProfileParser;
};
}
}

#include "nbl/builtin/hlsl/ies/texture.hlsl"

#endif // __NBL_ASSET_C_IES_PROFILE_H_INCLUDED__
