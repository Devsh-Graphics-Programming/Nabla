#ifndef _NBL_BUILTIN_HLSL_RWMC_RESOLVE_PARAMETERS_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_RWMC_RESOLVE_PARAMETERS_HLSL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

namespace nbl
{
namespace hlsl
{
namespace rwmc
{

struct ResolveParameters
{
	uint32_t lastCascadeIndex;
	float initialEmin; // a minimum image brightness that we always consider reliable
	float reciprocalBase;
	float reciprocalN;
	float reciprocalKappa;
	float colorReliabilityFactor;
	float NOverKappa;
};

ResolveParameters computeResolveParameters(float base, uint32_t sampleCount, float minReliableLuma, float kappa, uint32_t cascadeSize)
{
	ResolveParameters retval;
	retval.lastCascadeIndex = cascadeSize - 1u;
	retval.initialEmin = minReliableLuma;
	retval.reciprocalBase = 1.f / base;
	const float N = float(sampleCount);
	retval.reciprocalN = 1.f / N;
	retval.reciprocalKappa = 1.f / kappa;
	// if not interested in exact expected value estimation (kappa!=1.f), can usually accept a bit more variance relative to the image brightness we already have
	// allow up to ~<cascadeBase> more energy in one sample to lessen bias in some cases
	retval.colorReliabilityFactor = base + (1.f - base) * retval.reciprocalKappa;
	retval.NOverKappa = N * retval.reciprocalKappa;

	return retval;
}

}
}
}

#endif