#ifndef _NBL_BUILTIN_HLSL_RWMC_RWMC_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_RWMC_RWMC_HLSL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include <nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl>

namespace nbl
{
namespace hlsl
{
namespace rwmc
{
namespace impl
{

struct CascadeSample
{
	float32_t3 centerValue;
	float normalizedCenterLuma;
	float normalizedNeighbourhoodAverageLuma;
};

// TODO: figure out what values should pixels outside have, 0.0f is incorrect
float32_t3 sampleCascadeTexel(int32_t2 currentCoord, int32_t2 offset, in RWTexture2DArray<float32_t4> cascade, uint32_t cascadeIndex)
{
	const int32_t2 texelCoord = currentCoord + offset;
	if (any(texelCoord < int32_t2(0, 0)))
		return float32_t3(0.0f, 0.0f, 0.0f);

	float32_t4 output = cascade.Load(int32_t3(texelCoord, int32_t(cascadeIndex)));
	return float32_t3(output.r, output.g, output.b);
}

float32_t calcLuma(in float32_t3 col)
{
	return hlsl::dot<float32_t3>(hlsl::transpose(colorspace::scRGBtoXYZ)[1], col);
}

CascadeSample SampleCascade(in int32_t2 coord, in RWTexture2DArray<float32_t4> cascade, in uint cascadeIndex, in float reciprocalBaseI)
{
	float32_t3 neighbourhood[9];
	neighbourhood[0] = sampleCascadeTexel(coord, int32_t2(-1, -1), cascade, cascadeIndex);
	neighbourhood[1] = sampleCascadeTexel(coord, int32_t2(0, -1), cascade, cascadeIndex);
	neighbourhood[2] = sampleCascadeTexel(coord, int32_t2(1, -1), cascade, cascadeIndex);
	neighbourhood[3] = sampleCascadeTexel(coord, int32_t2(-1, 0), cascade, cascadeIndex);
	neighbourhood[4] = sampleCascadeTexel(coord, int32_t2(0, 0), cascade, cascadeIndex);
	neighbourhood[5] = sampleCascadeTexel(coord, int32_t2(1, 0), cascade, cascadeIndex);
	neighbourhood[6] = sampleCascadeTexel(coord, int32_t2(-1, 1), cascade, cascadeIndex);
	neighbourhood[7] = sampleCascadeTexel(coord, int32_t2(0, 1), cascade, cascadeIndex);
	neighbourhood[8] = sampleCascadeTexel(coord, int32_t2(1, 1), cascade, cascadeIndex);

	// numerical robustness
	float32_t3 excl_hood_sum = ((neighbourhood[0] + neighbourhood[1]) + (neighbourhood[2] + neighbourhood[3])) +
		((neighbourhood[5] + neighbourhood[6]) + (neighbourhood[7] + neighbourhood[8]));

	CascadeSample retval;
	retval.centerValue = neighbourhood[4];
	retval.normalizedNeighbourhoodAverageLuma = retval.normalizedCenterLuma = calcLuma(neighbourhood[4]) * reciprocalBaseI;
	retval.normalizedNeighbourhoodAverageLuma = (calcLuma(excl_hood_sum) * reciprocalBaseI + retval.normalizedNeighbourhoodAverageLuma) / 9.f;
	return retval;
}

} // namespace impl

struct ReweightingParameters
{
	uint32_t lastCascadeIndex;
	float initialEmin; // a minimum image brightness that we always consider reliable
	float reciprocalBase;
	float reciprocalN;
	float reciprocalKappa;
	float colorReliabilityFactor;
	float NOverKappa;
};

ReweightingParameters computeReweightingParameters(float base, uint32_t sampleCount, float minReliableLuma, float kappa, uint32_t cascadeSize)
{
	ReweightingParameters retval;
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

float32_t3 reweight(in ReweightingParameters params, in RWTexture2DArray<float32_t4> cascade, in int32_t2 coord)
{
	float reciprocalBaseI = 1.f;
	impl::CascadeSample curr = impl::SampleCascade(coord, cascade, 0u, reciprocalBaseI);

	float32_t3 accumulation = float32_t3(0.0f, 0.0f, 0.0f);
	float Emin = params.initialEmin;

	float prevNormalizedCenterLuma, prevNormalizedNeighbourhoodAverageLuma;
	for (uint i = 0u; i <= params.lastCascadeIndex; i++)
	{
		const bool notFirstCascade = i != 0u;
		const bool notLastCascade = i != params.lastCascadeIndex;

		impl::CascadeSample next;
		if (notLastCascade)
		{
			reciprocalBaseI *= params.reciprocalBase;
			next = impl::SampleCascade(coord, cascade, i + 1u, reciprocalBaseI);
		}

		float reliability = 1.f;
		// sample counting-based reliability estimation
		if (params.reciprocalKappa <= 1.f)
		{
			float localReliability = curr.normalizedCenterLuma;
			// reliability in 3x3 pixel block (see robustness)
			float globalReliability = curr.normalizedNeighbourhoodAverageLuma;
			if (notFirstCascade)
			{
				localReliability += prevNormalizedCenterLuma;
				globalReliability += prevNormalizedNeighbourhoodAverageLuma;
			}
			if (notLastCascade)
			{
				localReliability += next.normalizedCenterLuma;
				globalReliability += next.normalizedNeighbourhoodAverageLuma;
			}
			// check if above minimum sampling threshold (avg 9 sample occurences in 3x3 neighbourhood), then use per-pixel reliability (NOTE: tertiary op is in reverse)
			reliability = globalReliability < params.reciprocalN ? globalReliability : localReliability;
			{
				const float accumLuma = impl::calcLuma(accumulation);
				if (accumLuma > Emin)
					Emin = accumLuma;

				const float colorReliability = Emin * reciprocalBaseI * params.colorReliabilityFactor;

				reliability += colorReliability;
				reliability *= params.NOverKappa;
				reliability -= params.reciprocalKappa;
				reliability = clamp(reliability * 0.5f, 0.f, 1.f);
			}
		}
		accumulation += curr.centerValue * reliability;

		prevNormalizedCenterLuma = curr.normalizedCenterLuma;
		prevNormalizedNeighbourhoodAverageLuma = curr.normalizedNeighbourhoodAverageLuma;
		curr = next;
	}

	return accumulation;
}

}
}
}

#endif