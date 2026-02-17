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
NBL_CONCEPT_BEGIN(1)
#define a NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
NBL_CONCEPT_END(
	((NBL_CONCEPT_REQ_TYPE)(T::scalar_t))
	((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(concepts::FloatingPointScalar, typename T::scalar_t))
	((NBL_CONCEPT_REQ_EXPR_RET_TYPE)(
		(a.template calcLuma<SampleType, colorspace::scRGB>(::nbl::hlsl::experimental::declval<SampleType>())),
		::nbl::hlsl::is_same_v,
		typename T::scalar_t
	))
);
#undef a
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template<typename T, typename SampleType>
NBL_BOOL_CONCEPT ResolveLumaParams = ResolveLumaParamsBase<T, SampleType>;

template<typename T>
NBL_BOOL_CONCEPT ResolveAccessor = concepts::accessors::MipmappedLoadableImage<T, typename T::output_scalar_t, 2, T::Components>;

template<typename AccessorType, typename OutputScalar NBL_PRIMARY_REQUIRES(ResolveAccessor<AccessorType>)
struct SResolveAccessorAdaptor
{
	using output_scalar_t = OutputScalar;
	NBL_CONSTEXPR_STATIC_INLINE int32_t Components = 3;
	using output_t = vector<OutputScalar, Components>;
	NBL_CONSTEXPR_STATIC_INLINE int32_t image_dimension = 2;

	template<typename OutputScalarType, int32_t Dimension>
	void get(vector<uint16_t, 2> uv, uint16_t layer, uint16_t level, NBL_REF_ARG(output_t) value)
	{
		typename AccessorType::output_t sampled;
		accessor.template get<typename AccessorType::output_scalar_t, Dimension>(uv, layer, level, sampled);
		value = sampled.xyz;
	}

	template<typename OutputScalarType, int32_t Dimension>
	output_t get(vector<uint16_t, 2> uv, uint16_t layer, uint16_t level)
	{
		output_t value;
		get<OutputScalarType, Dimension>(uv, layer, level, value);
		return value;
	}

	AccessorType accessor;
};

template<typename CascadeAccessor, uint16_t CascadeCount NBL_PRIMARY_REQUIRES(
	ResolveAccessor<CascadeAccessor> &&
	ResolveLumaParams<SResolveParameters, typename CascadeAccessor::output_t>
)
struct SResolver
{
	using output_t = typename CascadeAccessor::output_t;
	using output_scalar_t = typename vector_traits<output_t>::scalar_type;
	using scalar_t = typename SResolveParameters::scalar_t;
	NBL_CONSTEXPR_STATIC_INLINE uint16_t last_cascade = uint16_t(CascadeCount - 1u);

	struct SCascadeSample
	{
		output_t centerValue;
		scalar_t normalizedCenterLuma;
		scalar_t normalizedNeighbourhoodAverageLuma;
	};

	static SResolver create(NBL_REF_ARG(SResolveParameters) resolveParameters)
	{
		SResolver retval;
		retval.params = resolveParameters;

		return retval;
	}

	output_t operator()(NBL_REF_ARG(CascadeAccessor) acc, const int16_t2 coord)
	{
		scalar_t reciprocalBaseI = 1.f;
		SCascadeSample curr = __sampleCascade(acc, coord, 0u, reciprocalBaseI);

		output_t accumulation = promote<output_t, scalar_t>(0.0f);
		scalar_t Emin = params.initialEmin;

		scalar_t prevNormalizedCenterLuma, prevNormalizedNeighbourhoodAverageLuma;
		NBL_UNROLL
		for (uint16_t i = 0u; i <= last_cascade; i++)
		{
			const bool notFirstCascade = i != 0;
			const bool notLastCascade = i != last_cascade;

			SCascadeSample next;
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

	SResolveParameters params;

	// pseudo private stuff:

	SCascadeSample __sampleCascade(NBL_REF_ARG(CascadeAccessor) acc, int16_t2 coord, uint16_t cascadeIndex, scalar_t reciprocalBaseI)
	{
		output_t sampleValue;
		scalar_t excl_hood_luma_sum = 0.f;

		acc.template get<output_scalar_t, 2>(coord + int16_t2(-1, -1), cascadeIndex, 0u, sampleValue);
		excl_hood_luma_sum += params.template calcLuma<output_t>(sampleValue);
		acc.template get<output_scalar_t, 2>(coord + int16_t2(0, -1), cascadeIndex, 0u, sampleValue);
		excl_hood_luma_sum += params.template calcLuma<output_t>(sampleValue);
		acc.template get<output_scalar_t, 2>(coord + int16_t2(1, -1), cascadeIndex, 0u, sampleValue);
		excl_hood_luma_sum += params.template calcLuma<output_t>(sampleValue);
		acc.template get<output_scalar_t, 2>(coord + int16_t2(-1, 0), cascadeIndex, 0u, sampleValue);
		excl_hood_luma_sum += params.template calcLuma<output_t>(sampleValue);

		SCascadeSample retval;
		acc.template get<output_scalar_t, 2>(coord + int16_t2(0, 0), cascadeIndex, 0u, retval.centerValue);
		const scalar_t centerLuma = params.template calcLuma<output_t>(retval.centerValue);
		acc.template get<output_scalar_t, 2>(coord + int16_t2(1, 0), cascadeIndex, 0u, sampleValue);
		excl_hood_luma_sum += params.template calcLuma<output_t>(sampleValue);
		acc.template get<output_scalar_t, 2>(coord + int16_t2(-1, 1), cascadeIndex, 0u, sampleValue);
		excl_hood_luma_sum += params.template calcLuma<output_t>(sampleValue);
		acc.template get<output_scalar_t, 2>(coord + int16_t2(0, 1), cascadeIndex, 0u, sampleValue);
		excl_hood_luma_sum += params.template calcLuma<output_t>(sampleValue);
		acc.template get<output_scalar_t, 2>(coord + int16_t2(1, 1), cascadeIndex, 0u, sampleValue);
		excl_hood_luma_sum += params.template calcLuma<output_t>(sampleValue);

		retval.normalizedCenterLuma = centerLuma * reciprocalBaseI;
		retval.normalizedNeighbourhoodAverageLuma = (excl_hood_luma_sum + centerLuma) * reciprocalBaseI / 9.f;
		return retval;
	}
};

}
}
}

#endif
