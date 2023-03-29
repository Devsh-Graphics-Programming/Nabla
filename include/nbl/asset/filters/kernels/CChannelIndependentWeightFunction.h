// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_CHANNEL_INDEPENDENT_IMAGE_FILTER_KERNEL_H_INCLUDED_
#define _NBL_ASSET_C_CHANNEL_INDEPENDENT_IMAGE_FILTER_KERNEL_H_INCLUDED_


#include "nbl/asset/filters/kernels/CFloatingPointSeparableImageFilterKernel.h"

#include <type_traits>
#include <tuple>


namespace nbl::asset
{

// we always invoke the first channel of the kernel for each kernel assigned to a channel
template <class FirstFunction, class... OtherFunctions>
class CChannelIndependentWeightFunction
{
	static_assert(sizeof...(OtherFunctions) < 4u);
	static inline constexpr size_t MaxChannels = 1 + sizeof...(OtherFunctions);

	constexpr static inline float min_supports[MaxChannels] = { FirstFunction::min_support,OtherFunctions::min_support... };
	constexpr static inline float max_supports[MaxChannels] = { FirstFunction::max_support,OtherFunctions::max_support... };
	constexpr static inline uint32_t _smoothnesses[MaxChannels] = { FirstFunction::k_smoothness,OtherFunctions::k_smoothness... };

	template<uint8_t ch>
	constexpr static inline bool has_kernel_v = ch < MaxChannels;

	using functions_t = std::tuple<FirstFunction, OtherFunctions...>;
	static_assert((std::is_same_v<impl::weight_function_value_type_t<FirstFunction>, impl::weight_function_value_type_t<OtherFunctions...>>), "Value Types neeed to be identical!");
	functions_t functions;

public:
	// TODO: should we upgrade the whole API to allow for per-channel supports?
	constexpr static inline float min_support = std::max_element(min_supports, min_supports + MaxChannels);
	constexpr static inline float max_support = std::min_element(max_supports, max_supports + MaxChannels);
	constexpr static inline uint32_t k_smoothness = std::min_element(_smoothnesses, _smoothnesses + MaxChannels);

	//
	CChannelIndependentWeightFunction(FirstFunction&& firstFunc, OtherFunctions&&... otherFuncs) : functions(std::move(firstFunc), std::move(otherFuncs)...) {}

	//	
	template <int32_t derivative>
	inline float operator()(float x, uint8_t channel) const
	{
		switch (channel)
		{
			// TODO: if the `std::get` fails to compile then use the weird `kernel_t` trait trick
		case 0:
			return std::get<0>().operator() < derivative > (x, 0);
		case 1:
			if constexpr (has_kernel_v<1>)
				return std::get<1>().operator() < derivative > (x, 1);
			break;
		case 2:
			if constexpr (has_kernel_v<2>)
				return std::get<2>().operator() < derivative > (x, 2);
			break;
		case 3:
			if constexpr (has_kernel_v<3>)
				return std::get<3>().operator() < derivative > (x, 3);
			break;
		default:
			break;
		}
		return core::nan<float>();
	}
};

} // end namespace nbl::asset

#endif