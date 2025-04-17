// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_LAMBERTIAN_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_LAMBERTIAN_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/sampling/cos_weighted.hlsl"
#include "nbl/builtin/hlsl/bxdf/reflection.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace transmission
{

template<class LS, class Iso, class Aniso, class Spectrum NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Iso> && surface_interactions::Anisotropic<Aniso>)
struct SLambertianBxDF
{
    using this_t = SLambertianBxDF<LS, Iso, Aniso, Spectrum>;
    using scalar_type = typename LS::scalar_type;
    using ray_dir_info_type = typename LS::ray_dir_info_type;
    using isotropic_type = Iso;
    using anisotropic_type = Aniso;
    using sample_type = LS;
    using spectral_type = Spectrum;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;
    using params_t = SBxDFParams<scalar_type>;

    static this_t create()
    {
        this_t retval;
        // nothing here, just keeping convention with others
        return retval;
    }

    static this_t create(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        return create();
    }

    void init(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        // do nothing
    }

    scalar_type __eval_pi_factored_out(scalar_type absNdotL)
    {
        return absNdotL;
    }

    scalar_type eval(NBL_CONST_REF_ARG(params_t) params)
    {
        return __eval_pi_factored_out(params.NdotL) * numbers::inv_pi<scalar_type> * 0.5;
    }

    sample_type generate_wo_clamps(NBL_CONST_REF_ARG(anisotropic_type) interaction, NBL_CONST_REF_ARG(vector<scalar_type, 3>) u)
    {
        ray_dir_info_type L;
        L.direction = sampling::ProjectedSphere<scalar_type>::generate(u);
        return sample_type::createFromTangentSpace(interaction.getTangentSpaceV(), L, interaction.getFromTangentSpace());
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_type) interaction, NBL_CONST_REF_ARG(vector<scalar_type, 3>) u)
    {
        return generate_wo_clamps(interaction, u);
    }

    sample_type generate(NBL_CONST_REF_ARG(isotropic_type) interaction, NBL_CONST_REF_ARG(vector<scalar_type, 3>) u)
    {
        return generate_wo_clamps(anisotropic_type::create(interaction), u);
    }

    scalar_type pdf(NBL_CONST_REF_ARG(params_t) params)
    {
        return sampling::ProjectedSphere<scalar_type>::pdf(params.NdotL);
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_t) params)
    {
        sampling::quotient_and_pdf<scalar_type, scalar_type> qp = sampling::ProjectedSphere<scalar_type>::template quotient_and_pdf(params.NdotL);
        return quotient_pdf_type::create(hlsl::promote<spectral_type>(qp.quotient), qp.pdf);
    }
};

}
}
}
}

#endif
