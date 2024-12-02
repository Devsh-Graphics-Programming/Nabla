// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_REFLECTION_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_REFLECTION_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#inclued "nbl/builtin/hlsl/sampling/cos_weighted.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace reflection
{

// still need these?
template<class LightSample, class Iso, class Aniso, class RayDirInfo, typename Scalar 
    NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso> && surface_interactions::Anisotropic<Aniso> && ray_dir_info::Basic<RayDirInfo> && is_scalar_v<Scalar>)
LightSample cos_generate(NBL_CONST_REF_ARG(Iso) interaction)
{
    return LightSample(interaction.V.reflect(interaction.N,interaction.NdotV),interaction.NdotV,interaction.N);
}
template<class LightSample, class Iso, class Aniso, class RayDirInfo, typename Scalar 
    NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso> && surface_interactions::Anisotropic<Aniso> && ray_dir_info::Basic<RayDirInfo> && is_scalar_v<Scalar>)
LightSample cos_generate(NBL_CONST_REF_ARG(Aniso) interaction)
{
    return LightSample(interaction.V.reflect(interaction.N,interaction.NdotV),interaction.NdotV,interaction.T,interaction.B,interaction.N);
}

// for information why we don't check the relation between `V` and `L` or `N` and `H`, see comments for `nbl::hlsl::transmission::cos_quotient_and_pdf`
template<typename SpectralBins, typename Pdf NBL_FUNC_REQUIRES(spectral_of<SpectralBins,Pdf> && is_floating_point_v<Pdf>)
quotient_and_pdf<SpectralBins, Pdf> cos_quotient_and_pdf()
{
    return quotient_and_pdf<SpectralBins, Pdf>::create(SpectralBins(1.f), numeric_limits<float>::infinity);
}


template<typename Scalar NBL_PRIMARY_REQUIRES(is_scalar_v<Scalar>)
struct SLambertianBxDF
{
    template<class LightSample, class Iso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso>)    // maybe put template in struct vs function?
    Scalar eval(LightSample _sample, Iso interaction)
    {
        return max(_sample.NdotL, 0.0) * numbers::inv_pi<Scalar>;
    }

    template<class LightSample, class Aniso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Aniso>)
    LightSample generate(Aniso interaction, vector<Scalar, 2> u)
    {
        vector<Scalar, 3> L = projected_hemisphere_generate<Scalar>(u);
        return LightSample::createTangentSpace(interaction.getTangentSpaceV(), L, interaction.getTangentFrame());
    }

    template<class LightSample, class Iso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso>)
    Scalar pdf(LightSample _sample, Iso interaction)
    {
        return projected_hemisphere_pdf(max(_sample.NdotL, 0.0));
    }

    // TODO: check math here
    template<typename SpectralBins, class LightSample, class Iso NBL_FUNC_REQUIRES(spectral_of<SpectralBins,Pdf> && Sample<LightSample> && surface_interactions::Anisotropic<Aniso>)
    quotient_and_pdf<SpectralBins, Scalar> quotient_and_pdf(LightSample _sample, Iso interaction)
    {
        Scalar pdf;
        Scalar q = projected_hemisphere_quotient_and_pdf<Scalar>(pdf, max(_sample.NdotL, 0.0));
        return quotient_and_pdf<SpectralBins, Pdf>::create(SpectralBins(q), pdf);
    }
};


template<typename Scalar NBL_PRIMARY_REQUIRES(is_scalar_v<Scalar>)
struct SOrenNayarBxDF
{
    using vector_t2 = vector<Scalar, 2>;

    Scalar rec_pi_factored_out_wo_clamps(Scalar VdotL, Scalar maxNdotL, Scalar maxNdotV)
    {
        Scalar A2 = A * 0.5;
        vector_t2 AB = vector_t2(1.0, 0.0) + vector_t2(-0.5, 0.45) * vector_t2(A2, A2) / vector_t2(A2 + 0.33, A2 + 0.09);
        Scalar C = 1.0 / max(maxNdotL, maxNdotV);

        Scalar cos_phi_sin_theta = max(VdotL - maxNdotL * maxNdotV, 0.0);
        return (AB.x + AB.y * cos_phi_sin_theta * C);
    }

    template<class LightSample, class Iso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso>)    // maybe put template in struct vs function?
    Scalar eval(LightSample _sample, Iso interaction)
    {
        return maxNdotL * numbers::inv_pi<Scalar> * rec_pi_factored_out_wo_clamps(_sample.VdotL, max(_sample.NdotL,0.0), max(interaction.NdotV,0.0));
    }

    template<class LightSample, class Aniso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Aniso>)
    LightSample generate(Aniso interaction, vector<Scalar, 2> u)
    {
        vector<Scalar, 3> L = projected_hemisphere_generate<Scalar>(u);
        return LightSample::createTangentSpace(interaction.getTangentSpaceV(), L, interaction.getTangentFrame());
    }

    template<class LightSample, class Iso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso>)
    Scalar pdf(LightSample _sample, Iso interaction)
    {
        return projected_hemisphere_pdf(max(_sample.NdotL, 0.0));
    }

    // TODO: check math here
    template<typename SpectralBins, class LightSample, class Iso NBL_FUNC_REQUIRES(spectral_of<SpectralBins,Pdf> && Sample<LightSample> && surface_interactions::Anisotropic<Aniso>)
    quotient_and_pdf<SpectralBins, Scalar> quotient_and_pdf(LightSample _sample, Iso interaction)
    {
        Scalar pdf;
        projected_hemisphere_quotient_and_pdf<Scalar>(pdf, max(_sample.NdotL, 0.0));
        Scalar q = rec_pi_factored_out_wo_clamps(_sample.VdotL, max(_sample.NdotL,0.0), max(interaction.NdotV,0.0));
        return quotient_and_pdf<SpectralBins, Pdf>::create(SpectralBins(q), pdf);
    }

    Scalar A;   // set A first before eval
};

}
}
}
}

#endif
