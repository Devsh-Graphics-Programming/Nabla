// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_BRDF_SPECULAR_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_BRDF_SPECULAR_COMMON_INCLUDED_

#include <nbl/builtin/hlsl/bxdf/common.hlsl>
#include <nbl/builtin/hlsl/bxdf/ndf/common.hlsl>

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace brdf
{
namespace specular
{

template <class fresnel_t, class ndf_t, class Sample, class Interaction, class MicrofacetCache>
struct CookTorrance : BxDFBase<float3, float, Sample, Interaction, MicrofacetCache>
{
    fresnel_t fresnel;
    ndf::ndf_traits<ndf_t> ndf;
};

template <class IncomingRayDirInfo, class fresnel_t, class ndf_t>
struct IsotropicCookTorrance : CookTorrance<fresnel_t, ndf_t, LightSample<IncomingRayDirInfo>, surface_interactions::Isotropic<IncomingRayDirInfo>, IsotropicMicrofacetCache>
{

};

template <class IncomingRayDirInfo, class fresnel_t, class ndf_t>
struct AnisotropicCookTorrance : CookTorrance<fresnel_t, ndf_t, LightSample<IncomingRayDirInfo>, surface_interactions::Anisotropic<IncomingRayDirInfo>, AnisotropicMicrofacetCache>
{

};

}
}
}
}
}

#endif
