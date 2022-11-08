// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_INCLUDED_

#include <nbl/builtin/hlsl/bxdf/common.hlsl>

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace transmission
{

template<class RayDirInfo>
LightSample<RayDirInfo> cos_generate(const surface_interactions::Isotropic<RayDirInfo> interaction)
{
  return LightSample<RayDirInfo>(interaction.V.transmit(),-1.f,interaction.N);
}
template<class RayDirInfo>
LightSample<RayDirInfo> cos_generate(const surface_interactions::Anisotropic<RayDirInfo> interaction)
{
  return LightSample<RayDirInfo>(interaction.V.transmit(),-1.f,interaction.T,interaction.B,interaction.N);
}

// Why don't we check that the incoming and outgoing directions equal each other
// (or similar for other delta distributions such as reflect, or smooth [thin] dielectrics):
// - The `quotient_and_pdf` functions are meant to be used with MIS and RIS
// - Our own generator can never pick an improbable path, so no checking necessary
// - For other generators the estimator will be `f_BSDF*f_Light*f_Visibility*clampedCos(theta)/(1+(p_BSDF^alpha+p_otherNonChosenGenerator^alpha+...)/p_ChosenGenerator^alpha)`
//	 therefore when `p_BSDF` equals `nbl_glsl_FLT_INF` it will drive the overall MIS estimator for the other generators to 0 so no checking necessary
template<typename SpectralBins>
quotient_and_pdf<SpectralBins> cos_quotient_and_pdf()
{
  return quotient_and_pdf<SpectralBins>::create(SpectralBins(1.f),nbl::hlsl::numeric_limits<float>::inf());
}

}
}
}
}

#endif
