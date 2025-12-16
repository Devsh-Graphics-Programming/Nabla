#ifndef _NBL_BUILTIN_HLSL_RWMC_RESOLVE_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_RWMC_RESOLVE_HLSL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include <nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl>
#include <nbl/builtin/hlsl/rwmc/ResolveParameters.hlsl>
#include <nbl/builtin/hlsl/concepts/accessors/loadable_image.hlsl>
#include <nbl/builtin/hlsl/colorspace.hlsl>
#include <nbl/builtin/hlsl/vector_utils/vector_traits.hlsl>

namespace nbl
{
namespace hlsl
{
namespace rwmc
{
		// declare concept
#define NBL_CONCEPT_NAME ResolveAccessorBase
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)(typename)(int32_t)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)(VectorScalarType)(Dims)
// not the greatest syntax but works
#define NBL_CONCEPT_PARAM_0 (a,T)
#define NBL_CONCEPT_PARAM_1 (scalar,VectorScalarType)
// start concept
	NBL_CONCEPT_BEGIN(2)
// need to be defined AFTER the concept begins
#define a NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define scalar NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
NBL_CONCEPT_END(
	((NBL_CONCEPT_REQ_EXPR)((a.calcLuma(vector<VectorScalarType, 3>(scalar, scalar, scalar)))))
);
#undef a
#undef scalar
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

/* ResolveAccessor is required to:
*	- satisfy `LoadableImage` concept requirements
*	- implement function called `calcLuma` which calculates luma from a 3 component pixel value
*/

template<typename T, typename VectorScalarType, int32_t Dims>
NBL_BOOL_CONCEPT ResolveAccessor = ResolveAccessorBase<T, VectorScalarType, Dims> && concepts::accessors::LoadableImage<T, VectorScalarType, Dims>;

template<typename CascadeAccessor, typename OutputColorTypeVec NBL_PRIMARY_REQUIRES(concepts::Vector<OutputColorTypeVec> && ResolveAccessor<CascadeAccessor, typename CascadeAccessor::output_scalar_type, CascadeAccessor::image_dimension>)
struct Resolver
{
	using output_type = OutputColorTypeVec;
	using scalar_t = typename vector_traits<output_type>::scalar_type;

	struct CascadeSample
	{
		float32_t3 centerValue;
		float normalizedCenterLuma;
		float normalizedNeighbourhoodAverageLuma;
	};

	static Resolver create(NBL_REF_ARG(ResolveParameters) resolveParameters)
	{
		Resolver retval;
		retval.params = resolveParameters;

		return retval;
	}

	output_type operator()(NBL_REF_ARG(CascadeAccessor) acc, const int16_t2 coord)
	{
		using scalar_t = typename vector_traits<output_type>::scalar_type;

		scalar_t reciprocalBaseI = 1.f;
		CascadeSample curr = __sampleCascade(acc, coord, 0u, reciprocalBaseI);

		output_type accumulation = output_type(0.0f, 0.0f, 0.0f);
		scalar_t Emin = params.initialEmin;

		scalar_t prevNormalizedCenterLuma, prevNormalizedNeighbourhoodAverageLuma;
		for (int16_t i = 0u; i <= params.lastCascadeIndex; i++)
		{
			const bool notFirstCascade = i != 0;
			const bool notLastCascade = i != params.lastCascadeIndex;

			CascadeSample next;
			if (notLastCascade)
			{
				reciprocalBaseI *= params.reciprocalBase;
				next = __sampleCascade(acc, coord, int16_t(i + 1), reciprocalBaseI);
			}

			scalar_t reliability = 1.f;
			// sample counting-based reliability estimation
			if (params.reciprocalKappa <= 1.f)
			{
				scalar_t localReliability = curr.normalizedCenterLuma;
				// reliability in 3x3 pixel block (see robustness)
				scalar_t globalReliability = curr.normalizedNeighbourhoodAverageLuma;
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
					const scalar_t accumLuma = acc.calcLuma(accumulation);
					if (accumLuma > Emin)
						Emin = accumLuma;

					const scalar_t colorReliability = Emin * reciprocalBaseI * params.colorReliabilityFactor;

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

	ResolveParameters params;

	// pseudo private stuff:

	CascadeSample __sampleCascade(NBL_REF_ARG(CascadeAccessor) acc, int16_t2 coord, uint16_t cascadeIndex, scalar_t reciprocalBaseI)
	{
		output_type neighbourhood[9];
		neighbourhood[0] = acc.template get<scalar_t, 2>(coord + int16_t2(-1, -1), cascadeIndex).xyz;
		neighbourhood[1] = acc.template get<scalar_t, 2>(coord + int16_t2(0, -1), cascadeIndex).xyz;
		neighbourhood[2] = acc.template get<scalar_t, 2>(coord + int16_t2(1, -1), cascadeIndex).xyz;
		neighbourhood[3] = acc.template get<scalar_t, 2>(coord + int16_t2(-1, 0), cascadeIndex).xyz;
		neighbourhood[4] = acc.template get<scalar_t, 2>(coord + int16_t2(0, 0), cascadeIndex).xyz;
		neighbourhood[5] = acc.template get<scalar_t, 2>(coord + int16_t2(1, 0), cascadeIndex).xyz;
		neighbourhood[6] = acc.template get<scalar_t, 2>(coord + int16_t2(-1, 1), cascadeIndex).xyz;
		neighbourhood[7] = acc.template get<scalar_t, 2>(coord + int16_t2(0, 1), cascadeIndex).xyz;
		neighbourhood[8] = acc.template get<scalar_t, 2>(coord + int16_t2(1, 1), cascadeIndex).xyz;

		// numerical robustness
		float32_t3 excl_hood_sum = ((neighbourhood[0] + neighbourhood[1]) + (neighbourhood[2] + neighbourhood[3])) +
			((neighbourhood[5] + neighbourhood[6]) + (neighbourhood[7] + neighbourhood[8]));

		CascadeSample retval;
		retval.centerValue = neighbourhood[4];
		retval.normalizedNeighbourhoodAverageLuma = retval.normalizedCenterLuma = acc.calcLuma(neighbourhood[4]) * reciprocalBaseI;
		retval.normalizedNeighbourhoodAverageLuma = (acc.calcLuma(excl_hood_sum) * reciprocalBaseI + retval.normalizedNeighbourhoodAverageLuma) / 9.f;
		return retval;
	}
};

}
}
}

#endif