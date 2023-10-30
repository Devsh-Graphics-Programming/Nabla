// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_CHANNEL_INDEPENDENT_IMAGE_FILTER_KERNEL_H_INCLUDED_
#define _NBL_ASSET_C_CHANNEL_INDEPENDENT_IMAGE_FILTER_KERNEL_H_INCLUDED_

#include <type_traits>
#include <tuple>


namespace nbl::asset
{

// we always invoke the first channel of the kernel for each kernel assigned to a channel
template <WeightFunction1D FirstWeightFunction1D, WeightFunction1D... OtherWeightFunctions>
class CChannelIndependentWeightFunction1D final
{
	public:
		using value_t = FirstWeightFunction1D::value_t;
		static_assert(std::is_same_v<value_t,double>,"should probably allow `float`s at some point!");
	
		static_assert(sizeof...(OtherWeightFunctions) < 4u);
		static inline constexpr size_t ChannelCount = 1 + sizeof...(OtherWeightFunctions);

	private:
		using functions_t = std::tuple<FirstWeightFunction1D,OtherWeightFunctions...>;
		static_assert((std::is_same_v<value_t,typename OtherWeightFunctions::value_t> && ...), "Value Types neeed to be identical!");
		functions_t functions;
		float m_minSupport = std::numeric_limits<float>::max();
		float m_maxSupport = std::numeric_limits<float>::min();
		uint32_t m_windowSize;

		// stuff needed for type deduction in `getFunction`
		template<uint8_t ch>
		constexpr static inline bool has_function_v = ch < ChannelCount;
		struct dummy_function_t {};
		template <uint8_t ch>
		using function_t = std::conditional_t<has_function_v<ch>, std::tuple_element_t<std::min<size_t>(static_cast<size_t>(ch), ChannelCount-1ull), functions_t>, dummy_function_t>;

	public:
		CChannelIndependentWeightFunction1D(FirstWeightFunction1D&& firstFunc, OtherWeightFunctions&&... otherFuncs) : functions(std::move(firstFunc), std::move(otherFuncs)...)
		{
			auto getMinMax = [this](const auto& element)
			{
				m_minSupport = core::min(m_minSupport, element.getMinSupport());
				m_maxSupport = core::max(m_maxSupport, element.getMaxSupport());
			};
			std::apply([&getMinMax](const auto&... elements) { (getMinMax(elements), ...); }, functions);
			
			// The reason we use a ceil for window_size:
			// For a convolution operation, depending upon where you place the kernel center in the output image it can encompass different number of input pixel centers.
			// For example, assume you have a 1D kernel with supports [-3/4, 3/4) and you place this at x=0.5, then kernel weights will be
			// non-zero in [-3/4 + 0.5, 3/4 + 0.5) so there will be only one pixel center (at x=0.5) in the non-zero kernel domain, hence window_size will be 1.
			// But if you place the same kernel at x=0, then the non-zero kernel domain will become [-3/4, 3/4) which now encompasses two pixel centers
			// (x=-0.5 and x=0.5), that is window_size will be 2.
			// Note that the window_size can never exceed 2, in the above case, because for that to happen there should be more than 2 pixel centers in non-zero
			// kernel domain which is not possible given that two pixel centers are always separated by a distance of 1.
			m_windowSize = static_cast<int32_t>(core::ceil<float>(m_maxSupport-m_minSupport));
		}

		template <uint8_t ch>
		const function_t<ch>& getFunction() const { return std::get<static_cast<size_t>(ch)>(functions); }

		inline value_t weight(const float x, const uint8_t channel) const
		{
			switch (channel)
			{
				case 0:
					return getFunction<0>().weight(x);
				case 1:
					if constexpr (has_function_v<1>)
						return getFunction<1>().weight(x);
					break;
				case 2:
					if constexpr (has_function_v<2>)
						return getFunction<2>().weight(x);
					break;
				case 3:
					if constexpr (has_function_v<3>)
						return getFunction<3>().weight(x);
					break;
				default:
					break;
			}
			return core::nan<float>();
		}

		inline float getMinSupport() const { return m_minSupport; }
		inline float getMaxSupport() const { return m_maxSupport; }

		// given an unnormalized (measured in pixels), center sampled (sample at the center of the pixel) coordinate (origin is at the center of the first pixel),
		// return corner sampled coordinate (origin at the very edge of the first pixel) as well as the
		// corner sampled coordinate of the first pixel that lays inside the kernel's support when centered on the given pixel
		inline int32_t getWindowMinCoord(const float unnormCenterSampledCoord, float& cornerSampledCoord) const
		{
			cornerSampledCoord = unnormCenterSampledCoord - 0.5f;
			return static_cast<int32_t>(core::ceil<float>(cornerSampledCoord + m_minSupport));
		}

		// overload that does not return the cornern sampled coordinate of the given center sampled coordinate
		inline int32_t getWindowMinCoord(const float unnormCeterSampledCoord) const
		{
			float dummy;
			return getWindowMinCoord(unnormCeterSampledCoord, dummy);
		}

		// get the kernel support (measured in pixels)
		inline int32_t getWindowSize() const { return m_windowSize; }

		// TODO: Do we even need to keep this function around!?
		static inline bool validate(ICPUImage* inImage, ICPUImage* outImage)
		{
			const auto& inParams = inImage->getCreationParameters();
			const auto& outParams = inImage->getCreationParameters();
			return !(isIntegerFormat(inParams.format)||isIntegerFormat(outParams.format));
		}
};

template<WeightFunction1D R, WeightFunction1D G = R, WeightFunction1D B = G, WeightFunction1D A = B>
using CDefaultChannelIndependentWeightFunction1D = CChannelIndependentWeightFunction1D<R, G, B, A>;

} // end namespace nbl::asset

#endif
