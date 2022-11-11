// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_REFLECTION_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_REFLECTION_INCLUDED_

#include <nbl/builtin/hlsl/bxdf/common.hlsl>

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace reflection
{

template<class RayDirInfo>
LightSample<RayDirInfo> cos_generate(const surface_interactions::Isotropic<RayDirInfo> interaction)
{
  return LightSample<RayDirInfo>(interaction.V.reflect(interaction.N,interaction.NdotV),interaction.NdotV,interaction.N);
}
template<class RayDirInfo>
LightSample<RayDirInfo> cos_generate(const surface_interactions::Anisotropic<RayDirInfo> interaction)
{
  return LightSample<RayDirInfo>(interaction.V.reflect(interaction.N,interaction.NdotV),interaction.NdotV,interaction.T,interaction.B,interaction.N);
}

// for information why we don't check the relation between `V` and `L` or `N` and `H`, see comments for `nbl::hlsl::transmission::cos_quotient_and_pdf`
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
