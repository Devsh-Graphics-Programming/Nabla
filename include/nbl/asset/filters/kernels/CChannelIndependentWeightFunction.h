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
template <class FirstWeightFunction1D, class... OtherWeightFunctions>
class CChannelIndependentWeightFunction
{
	static_assert(sizeof...(OtherWeightFunctions) < 4u);
	static inline constexpr size_t MaxChannels = 1 + sizeof...(OtherWeightFunctions);

	constexpr static inline uint32_t _smoothnesses[MaxChannels] = { FirstWeightFunction1D::k_smoothness,OtherWeightFunctions::k_smoothness... };

	template<uint8_t ch>
	constexpr static inline bool has_function_v = ch < MaxChannels;

	using functions_t = std::tuple<FirstWeightFunction1D, OtherWeightFunctions...>;
	static_assert((std::is_same_v<impl::weight_function_value_type_t<FirstWeightFunction1D>, impl::weight_function_value_type_t<OtherWeightFunctions...>>), "Value Types neeed to be identical!");
	functions_t functions;

	struct dummy_function_t {};
	template <uint8_t ch>
	using function_t = std::conditional_t<has_function_v<ch>, std::tuple_element_t<std::min<size_t>(static_cast<size_t>(ch), MaxChannels-1ull), functions_t>, dummy_function_t>;

	template <uint8_t ch>
	const function_t<ch>& getFunction() const { return std::get<static_cast<size_t>(ch)>(functions); }

public:
	constexpr static inline uint32_t k_smoothness = std::min_element(_smoothnesses, _smoothnesses + MaxChannels)[0];
	constexpr static inline float k_energy[4] = { 0.f, 0.f, 0.f, 0.f }; // TODO(achal): Implement.

	CChannelIndependentWeightFunction(FirstWeightFunction1D&& firstFunc, OtherWeightFunctions&&... otherFuncs) : functions(std::move(firstFunc), std::move(otherFuncs)...)
	{
		updateSupports();
	}

	inline float operator()(float x, uint8_t channel) const
	{
		switch (channel)
		{
		case 0:
			return getFunction<0>().operator()(x, 0);
		case 1:
			if constexpr (has_function_v<1>)
				return getFunction<1>().operator()(x, 1);
			break;
		case 2:
			if constexpr (has_function_v<2>)
				return getFunction<2>().operator()(x, 2);
			break;
		case 3:
			if constexpr (has_function_v<3>)
				return getFunction<3>().operator()(x, 3);
			break;
		default:
			break;
		}
		return core::nan<float>();
	}

	inline void stretch(const float s)
	{
		auto stretch_ = [s](auto& element) {element.stretch(s); };
		std::apply([&stretch_](auto&... elements) { (stretch_(elements), ...); }, functions);

		updateSupports();
	}

	inline void scale(const float s)
	{
		auto scale_ = [s](auto& element) {element.scale(s); };
		std::apply([&scale_](auto&... elements) { (scale_(elements), ...); }, functions);
	}

	inline void stretchAndScale(const float stretchFactor)
	{
		stretch(stretchFactor);
		scale(1.f / stretchFactor);
	}

	inline float getMinSupport() const { return m_minSupport; }
	inline float getMaxSupport() const { return m_maxSupport; }
	inline float getInvStretch(const uint32_t channel = 0) const
	{
		switch (channel)
		{
		case 0:
			return getFunction<0>().getInvStretch();
		case 1:
			if constexpr (has_function_v<1>)
                return getFunction<1>().getInvStretch();
            break;
		case 2:
			if constexpr (has_function_v<2>)
                return getFunction<2>().getInvStretch();
            break;
		case 3:
			if constexpr (has_function_v<3>)
                return getFunction<3>().getInvStretch();
            break;
		default:
			break;
		}
		return core::nan<float>();
	}

private:
	inline void updateSupports()
	{
		auto getMinMax = [this](const auto& element)
		{
			m_minSupport = core::min(m_minSupport, element.getMinSupport());
			m_maxSupport = core::max(m_maxSupport, element.getMaxSupport());
		};
		std::apply([&getMinMax](const auto&... elements) { (getMinMax(elements), ...); }, functions);
	}

	float m_minSupport = std::numeric_limits<float>::max();
	float m_maxSupport = std::numeric_limits<float>::min();
};

} // end namespace nbl::asset

#endif