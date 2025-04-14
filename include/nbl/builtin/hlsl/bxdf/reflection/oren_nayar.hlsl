// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_REFLECTION_OREN_NAYAR_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_REFLECTION_OREN_NAYAR_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/sampling/cos_weighted.hlsl"
#include "nbl/builtin/hlsl/bxdf/geom_smith.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace reflection
{

template<class LightSample, class Iso, class Aniso, class Spectrum NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso> && surface_interactions::Anisotropic<Aniso>)
struct SOrenNayarBxDF
{
    using this_t = SOrenNayarBxDF<LightSample, Iso, Aniso, Spectrum>;
    using scalar_type = typename LightSample::scalar_type;
    using vector2_type = vector<scalar_type, 2>;
    using ray_dir_info_type = typename LightSample::ray_dir_info_type;

    using isotropic_type = Iso;
    using anisotropic_type = Aniso;
    using sample_type = LightSample;
    using spectral_type = Spectrum;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;
    using params_t = SBxDFParams<scalar_type>;

    static this_t create(scalar_type A)
    {
        this_t retval;
        retval.A = A;
        return retval;
    }

    static this_t create(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        return create(params.A.x);
    }

    void init(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        A = params.A.x;
    }

    scalar_type __rec_pi_factored_out_wo_clamps(scalar_type VdotL, scalar_type maxNdotL, scalar_type maxNdotV)
    {
        scalar_type A2 = A * 0.5;
        vector2_type AB = vector2_type(1.0, 0.0) + vector2_type(-0.5, 0.45) * vector2_type(A2, A2) / vector2_type(A2 + 0.33, A2 + 0.09);
        scalar_type C = 1.0 / max<scalar_type>(maxNdotL, maxNdotV);

        scalar_type cos_phi_sin_theta = max<scalar_type>(VdotL - maxNdotL * maxNdotV, 0.0);
        return (AB.x + AB.y * cos_phi_sin_theta * C);
    }

    scalar_type eval(NBL_CONST_REF_ARG(params_t) params)
    {
        return params.NdotL * numbers::inv_pi<scalar_type> * __rec_pi_factored_out_wo_clamps(params.VdotL, params.NdotL, params.NdotV);
    }

    sample_type generate_wo_clamps(NBL_CONST_REF_ARG(anisotropic_type) interaction, NBL_CONST_REF_ARG(vector2_type) u)
    {
        ray_dir_info_type L;
        L.direction = sampling::ProjectedHemisphere<scalar_type>::generate(u);
        return sample_type::createFromTangentSpace(interaction.getTangentSpaceV(), L, interaction.getFromTangentSpace());
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_type) interaction, NBL_CONST_REF_ARG(vector2_type) u)
    {
        return generate_wo_clamps(interaction, u);
    }

    scalar_type pdf(NBL_CONST_REF_ARG(params_t) params)
    {
        return sampling::ProjectedHemisphere<scalar_type>::pdf(params.NdotL);
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_t) params)
    {
        scalar_type _pdf = pdf(params);
        scalar_type q = __rec_pi_factored_out_wo_clamps(params.VdotL, params.NdotL, params.NdotV);
        return quotient_pdf_type::create(hlsl::promote<spectral_type>(q), _pdf);
    }

    scalar_type A;
};

}
}
}
}

#endif
