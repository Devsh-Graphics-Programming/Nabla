#ifndef _NBL_BUILTIN_HLSL_RWMC_RESOLVE_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_RWMC_RESOLVE_HLSL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include <nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl>
#include <nbl/builtin/hlsl/rwmc/ResolveParameters.hlsl>
#include <nbl/builtin/hlsl/concepts/accessors/loadable_image.hlsl>

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
#define NBL_CONCEPT_PARAM_2 (vec,vector<VectorScalarType, Dims>)
// start concept
	NBL_CONCEPT_BEGIN(2)
// need to be defined AFTER the concept begins
#define a NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define scalar NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define vec NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
	((NBL_CONCEPT_REQ_EXPR)((a.calcLuma(vec))))
);
#undef a
#undef vec
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

/* ResolveAccessor is required to:
*	- satisfy `LoadableImage` concept requirements
*	- implement function called `calcLuma` which calculates luma from a pixel value
*/

template<typename T, typename VectorScalarType, int32_t Dims>
NBL_BOOL_CONCEPT ResolveAccessor = ResolveAccessorBase<T, VectorScalarType, Dims> && concepts::accessors::LoadableImage<T, VectorScalarType, Dims>;

template<typename OutputScalar>
struct ResolveAccessorAdaptor
{
	using output_scalar_type = OutputScalar;
	using output_type = vector<OutputScalar, 4>;
	NBL_CONSTEXPR int32_t image_dimension = 2;

	RWTexture2DArray<float32_t4> cascade;

	float32_t calcLuma(in float32_t3 col)
	{
		return hlsl::dot<float32_t3>(hlsl::transpose(colorspace::scRGBtoXYZ)[1], col);
	}

	template<typename OutputScalarType, int32_t Dimension>
	output_type get(vector<uint16_t, 2> uv, uint16_t layer)
	{
		uint32_t imgWidth, imgHeight, layers;
		cascade.GetDimensions(imgWidth, imgHeight, layers);
		int16_t2 cascadeImageDimension = int16_t2(imgWidth, imgHeight);

		if (any(uv < int16_t2(0, 0)) || any(uv > cascadeImageDimension))
			return vector<OutputScalar, 4>(0, 0, 0, 0);

		return cascade.Load(int32_t3(uv, int32_t(layer)));
	}
};

template<typename CascadeAccessor, typename OutputColorType> //NBL_PRIMARY_REQUIRES(ResolveAccessor<CascadeAccessor, typename CascadeAccessor::output_scalar_type, CascadeAccessor::image_dimension>)
struct Resolver
{
	using output_type = OutputColorType;

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
		float reciprocalBaseI = 1.f;
		CascadeSample curr = __sampleCascade(acc, coord, 0u, reciprocalBaseI);

		float32_t3 accumulation = float32_t3(0.0f, 0.0f, 0.0f);
		float Emin = params.initialEmin;

		float prevNormalizedCenterLuma, prevNormalizedNeighbourhoodAverageLuma;
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
					const float accumLuma = acc.calcLuma(accumulation);
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

	ResolveParameters params;

	// pseudo private stuff:

	CascadeSample __sampleCascade(NBL_REF_ARG(CascadeAccessor) acc, int16_t2 coord, uint16_t cascadeIndex, float reciprocalBaseI)
	{
		typename CascadeAccessor::output_type tmp;
		output_type neighbourhood[9];
		neighbourhood[0] = acc.template get<float, 2>(coord + int16_t2(-1, -1), cascadeIndex);
		neighbourhood[1] = acc.template get<float, 2>(coord + int16_t2(0, -1), cascadeIndex);
		neighbourhood[2] = acc.template get<float, 2>(coord + int16_t2(1, -1), cascadeIndex);
		neighbourhood[3] = acc.template get<float, 2>(coord + int16_t2(-1, 0), cascadeIndex);
		neighbourhood[4] = acc.template get<float, 2>(coord + int16_t2(0, 0), cascadeIndex);
		neighbourhood[5] = acc.template get<float, 2>(coord + int16_t2(1, 0), cascadeIndex);
		neighbourhood[6] = acc.template get<float, 2>(coord + int16_t2(-1, 1), cascadeIndex);
		neighbourhood[7] = acc.template get<float, 2>(coord + int16_t2(0, 1), cascadeIndex);
		neighbourhood[8] = acc.template get<float, 2>(coord + int16_t2(1, 1), cascadeIndex);

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