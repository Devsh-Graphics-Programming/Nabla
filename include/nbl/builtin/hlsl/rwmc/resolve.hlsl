#ifndef _NBL_BUILTIN_HLSL_RWMC_RESOLVE_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_RWMC_RESOLVE_HLSL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include <nbl/builtin/hlsl/rwmc/ResolveParameters.hlsl>
#include <nbl/builtin/hlsl/concepts/accessors/loadable_image.hlsl>
#include <nbl/builtin/hlsl/vector_utils/vector_traits.hlsl>

namespace nbl
{
namespace hlsl
{
namespace rwmc
{
// declare concept
#define NBL_CONCEPT_NAME ResolveLumaParamsBase
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)(typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)(SampleType)
#define NBL_CONCEPT_PARAM_0 (a,T)
#define NBL_CONCEPT_PARAM_1 (sample,SampleType)
NBL_CONCEPT_BEGIN(2)
#define a NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define sample NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
NBL_CONCEPT_END(
	((NBL_CONCEPT_REQ_TYPE)(T::scalar_t))
	((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(concepts::FloatingPointScalar, T::scalar_t))
	((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((a.template calcLuma<SampleType>(sample)), ::nbl::hlsl::is_same_v, typename T::scalar_t))
);
#undef sample
#undef a
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template<typename T, typename SampleType>
NBL_BOOL_CONCEPT ResolveLumaParams = ResolveLumaParamsBase<T, SampleType>;

// declare concept
#define NBL_CONCEPT_NAME ResolveAccessorBase
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (a,T)
#define NBL_CONCEPT_PARAM_1 (uv,vector<uint16_t,2>)
#define NBL_CONCEPT_PARAM_2 (layer,uint16_t)
NBL_CONCEPT_BEGIN(3)
#define a NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define uv NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define layer NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
	((NBL_CONCEPT_REQ_TYPE)(T::output_t))
	((NBL_CONCEPT_REQ_TYPE)(T::output_scalar_t))
	((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((a.template get<typename T::output_scalar_t,2>(uv,layer)), ::nbl::hlsl::is_same_v, typename T::output_t))
);
#undef layer
#undef uv
#undef a
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template<typename T>
NBL_BOOL_CONCEPT ResolveAccessor = ResolveAccessorBase<T> && concepts::accessors::LoadableImage<T, typename T::output_scalar_t, 2, T::Components>;

template<typename OutputScalar>
struct ResolveAccessorAdaptor
{
	using output_scalar_t = OutputScalar;
	NBL_CONSTEXPR int32_t Components = 3;
	using output_t = vector<OutputScalar, Components>;
	NBL_CONSTEXPR int32_t image_dimension = 2;

	RWTexture2DArray<float32_t4> cascade;

	template<typename OutputScalarType, int32_t Dimension>
	output_t get(vector<uint16_t, 2> uv, uint16_t layer)
	{
		uint32_t imgWidth, imgHeight, layers;
		cascade.GetDimensions(imgWidth, imgHeight, layers);
		int16_t2 cascadeImageDimension = int16_t2(imgWidth, imgHeight);

		if (any(uv < int16_t2(0, 0)) || any(uv >= cascadeImageDimension))
			return promote<output_t, output_scalar_t>(0);

		return cascade.Load(int32_t3(uv, int32_t(layer)));
	}
};

template<typename CascadeAccessor, uint16_t CascadeCount, typename ResolveParamsType = ResolveParameters NBL_PRIMARY_REQUIRES(ResolveAccessor<CascadeAccessor> && ResolveLumaParams<ResolveParamsType, typename CascadeAccessor::output_t>)
struct Resolver
{
	using output_t = typename CascadeAccessor::output_t;
	using output_scalar_t = typename vector_traits<output_t>::scalar_type;
	using scalar_t = typename ResolveParamsType::scalar_t;
	NBL_CONSTEXPR static uint16_t last_cascade = CascadeCount - 1u;

	struct CascadeSample
	{
		output_t centerValue;
		float normalizedCenterLuma;
		float normalizedNeighbourhoodAverageLuma;
	};

	static Resolver create(NBL_REF_ARG(ResolveParamsType) resolveParameters)
	{
		Resolver retval;
		retval.params = resolveParameters;

		return retval;
	}

	output_t operator()(NBL_REF_ARG(CascadeAccessor) acc, const int16_t2 coord)
	{
		scalar_t reciprocalBaseI = 1.f;
		CascadeSample curr = __sampleCascade(acc, coord, 0u, reciprocalBaseI);

		output_t accumulation = promote<output_t, scalar_t>(0.0f);
		scalar_t Emin = params.initialEmin;

		scalar_t prevNormalizedCenterLuma, prevNormalizedNeighbourhoodAverageLuma;
		NBL_UNROLL
		for (uint16_t i = 0u; i <= last_cascade; i++)
		{
			const bool notFirstCascade = i != 0;
			const bool notLastCascade = i != last_cascade;

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
					const scalar_t accumLuma = params.template calcLuma<output_t>(accumulation);
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

	ResolveParamsType params;

	// pseudo private stuff:

	CascadeSample __sampleCascade(NBL_REF_ARG(CascadeAccessor) acc, int16_t2 coord, uint16_t cascadeIndex, scalar_t reciprocalBaseI)
	{
		output_t neighbourhood[9];
		neighbourhood[0] = acc.template get<output_scalar_t, 2>(coord + int16_t2(-1, -1), cascadeIndex);
		neighbourhood[1] = acc.template get<output_scalar_t, 2>(coord + int16_t2(0, -1), cascadeIndex);
		neighbourhood[2] = acc.template get<output_scalar_t, 2>(coord + int16_t2(1, -1), cascadeIndex);
		neighbourhood[3] = acc.template get<output_scalar_t, 2>(coord + int16_t2(-1, 0), cascadeIndex);
		neighbourhood[4] = acc.template get<output_scalar_t, 2>(coord + int16_t2(0, 0), cascadeIndex);
		neighbourhood[5] = acc.template get<output_scalar_t, 2>(coord + int16_t2(1, 0), cascadeIndex);
		neighbourhood[6] = acc.template get<output_scalar_t, 2>(coord + int16_t2(-1, 1), cascadeIndex);
		neighbourhood[7] = acc.template get<output_scalar_t, 2>(coord + int16_t2(0, 1), cascadeIndex);
		neighbourhood[8] = acc.template get<output_scalar_t, 2>(coord + int16_t2(1, 1), cascadeIndex);

		// numerical robustness
		output_t excl_hood_sum = ((neighbourhood[0] + neighbourhood[1]) + (neighbourhood[2] + neighbourhood[3])) +
			((neighbourhood[5] + neighbourhood[6]) + (neighbourhood[7] + neighbourhood[8]));

		CascadeSample retval;
		retval.centerValue = neighbourhood[4];
		retval.normalizedNeighbourhoodAverageLuma = retval.normalizedCenterLuma = params.template calcLuma<output_t>(neighbourhood[4]) * reciprocalBaseI;
		retval.normalizedNeighbourhoodAverageLuma = (params.template calcLuma<output_t>(excl_hood_sum) * reciprocalBaseI + retval.normalizedNeighbourhoodAverageLuma) / 9.f;
		return retval;
	}
};

}
}
}

#endif
