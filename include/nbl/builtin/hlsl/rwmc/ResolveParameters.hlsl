#ifndef _NBL_BUILTIN_HLSL_RWMC_RESOLVE_PARAMETERS_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_RWMC_RESOLVE_PARAMETERS_HLSL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include <nbl/builtin/hlsl/colorspace.hlsl>

namespace nbl
{
namespace hlsl
{
namespace rwmc
{

struct SResolveParameters
{
	using scalar_t = float32_t;

	static SResolveParameters create(scalar_t base, uint32_t sampleCount, scalar_t minReliableLuma, scalar_t kappa)
	{
		SResolveParameters retval;
		retval.initialEmin = minReliableLuma;
		retval.reciprocalBase = 1.f / base;
		const scalar_t N = scalar_t(sampleCount);
		retval.reciprocalN = 1.f / N;
		retval.reciprocalKappa = 1.f / kappa;
		// if not interested in exact expected value estimation (kappa!=1.f), can usually accept a bit more variance relative to the image brightness we already have
		// allow up to ~<cascadeBase> more energy in one sample to lessen bias in some cases
		retval.colorReliabilityFactor = base + (1.f - base) * retval.reciprocalKappa;
		retval.NOverKappa = N * retval.reciprocalKappa;

		return retval;
	}

	template<typename SampleType, typename Colorspace = colorspace::scRGB>
	scalar_t calcLuma(NBL_CONST_REF_ARG(SampleType) col)
	{
		return hlsl::dot<SampleType>(hlsl::transpose(Colorspace::ToXYZ())[1], col);
	}

	scalar_t initialEmin; // a minimum image brightness that we always consider reliable
	scalar_t reciprocalBase;
	scalar_t reciprocalN;
	scalar_t reciprocalKappa;
	scalar_t colorReliabilityFactor;
	scalar_t NOverKappa;
};

}
}
}

#endif
