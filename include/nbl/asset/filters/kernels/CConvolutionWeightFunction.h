// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_CONVOLUTION_WEIGHT_FUNCTION_H_INCLUDED_
#define _NBL_ASSET_C_CONVOLUTION_WEIGHT_FUNCTION_H_INCLUDED_

#include "nbl/asset/filters/kernels/WeightFunctions.h"

namespace nbl::asset
{

template<typename WeightFunction1DA, typename WeightFunction1DB>
class CConvolutionWeightFunction1D;

namespace impl
{

template <typename WeightFunction1DA, typename WeightFunction1DB>
struct convolution_weight_function_helper
{
	template <int32_t derivative>
	inline static float operator_impl(const CConvolutionWeightFunction1D<WeightFunction1DA, WeightFunction1DB>& _this, const float x, const uint32_t channel, const uint32_t sampleCount)
	{
		if constexpr (std::is_same_v<WeightFunction1DB, CWeightFunction1D<SDiracFunction>>)
		{
			return _this.m_funcA.operator()<derivative>(x, channel);
		}
		else if (std::is_same_v<WeightFunction1DA, CWeightFunction1D<SDiracFunction>>)
		{
			return _this.m_funcB.operator()<derivative>(x, channel);
		}
		else
		{
			constexpr auto deriv_A = std::min(static_cast<int32_t>(WeightFunction1DA::k_smoothness), derivative);
			constexpr auto deriv_B = derivative - deriv_A;

			auto [minIntegrationLimit, maxIntegrationLimit] = _this.getIntegrationDomain(x);
			// if this happens, it means that `m_ratio=INF` and it degenerated into a dirac delta
			if (minIntegrationLimit == maxIntegrationLimit)
			{
				assert(WeightFunction1DB::k_energy[channel] != 0.f);
				return _this.m_funcA.operator()<derivative>(x, channel) * WeightFunction1DB::k_energy[channel];
			}

			// if this happened then `m_ratio=0` and we have infinite domain, this is not a problem
			const double dt = (maxIntegrationLimit - minIntegrationLimit) / sampleCount;
			if (core::isnan<double>(dt))
				return _this.m_funcA.operator()<deriv_A>(x, channel) * _this.m_funcB.operator()<deriv_B>(0.f, channel);

			const auto ratio = _this.m_funcA.getInvStretch() / _this.m_funcB.getInvStretch(); // TODO(achal): Check this again!
			double result = 0.0;
			for (uint32_t i = 0u; i < sampleCount; ++i)
			{
				const double t = minIntegrationLimit + i * dt;
				result += _this.m_funcA.operator()<deriv_A>(x - t, channel) * _this.m_funcB.operator()<deriv_B>(t * ratio, channel) * dt;
			}
			return static_cast<float>(result);
		}
	}
};

// TODO: redo all to account for `m_ratio`

template <>
struct convolution_weight_function_helper<CWeightFunction1D<SBoxFunction>, CWeightFunction1D<SBoxFunction>>
{
	template <int32_t derivative>
	static float operator_impl(const CConvolutionWeightFunction1D<CWeightFunction1D<SBoxFunction>, CWeightFunction1D<SBoxFunction>>& _this, const float x, const uint32_t channel, const uint32_t sampleCount);
};

template <>
struct convolution_weight_function_helper<CWeightFunction1D<SGaussianFunction>, CWeightFunction1D<SGaussianFunction>>
{
	template <int32_t derivative>
	static float operator_impl(const CConvolutionWeightFunction1D<CWeightFunction1D<SGaussianFunction>, CWeightFunction1D<SGaussianFunction>>& _this, const float x, const uint32_t channel, const uint32_t sampleCount);
};

// TODO: Specialization: convolution_weight_function_helper<Triangle,Triangle> = this is tricky but feasible

template <>
struct convolution_weight_function_helper<CWeightFunction1D<SKaiserFunction>, CWeightFunction1D<SKaiserFunction>>
{
	template <int32_t derivative>
	static float operator_impl(const CConvolutionWeightFunction1D<CWeightFunction1D<SKaiserFunction>, CWeightFunction1D<SKaiserFunction>>& _this, const float x, const uint32_t channel, const uint32_t sampleCount);
};

} // end namespace impl

// TODO(achal): I can also make a concept for WeightFunction1D here so that it only allows the type CWeightFunction1D.

// this is the horribly slow generic version that you should not use (only use the specializations or when one of the weights is a dirac)
template<typename WeightFunction1DA, typename WeightFunction1DB>
class CConvolutionWeightFunction1D
{
	static_assert(std::is_same_v<impl::weight_function_value_type_t<WeightFunction1DA>, impl::weight_function_value_type_t<WeightFunction1DB>>, "Both functions must use the same Value Type!");

	friend struct impl::convolution_weight_function_helper<WeightFunction1DA, WeightFunction1DB>;
	const WeightFunction1DA m_funcA;
	const WeightFunction1DB m_funcB;

	std::pair<double, double> getIntegrationDomain(const float x) const
	{
		// TODO(achal): Why would I need their widths?
		// constexpr float WidthA = WeightFunction1DA::max_support - WeightFunction1DA::min_support;
		// const float WidthB = (WeightFunction1DB::max_support - WeightFunction1DB::min_support) / m_ratio;

		double minIntegrationLimit = 1.0;
		double maxIntegrationLimit = 0.0;

		// TODO: redo to account for `m_ratio`
		assert(minIntegrationLimit <= maxIntegrationLimit);

		return { minIntegrationLimit, maxIntegrationLimit };
	}

public:
	constexpr static inline uint32_t k_smoothness = WeightFunction1DA::k_smoothness + WeightFunction1DB::k_smoothness;

	// `_ratio` is the width ratio between kernel A and B, our operator() computes `a(x) \conv b(x*_ratio)`
	// if you want to compute `f(x) = a(x/c_1) \conv b(x/c_2)` then you can compute `f(x) = c_1 g(u)` where `u=x/c_1` and `_ratio = c_1/c_2`
	// so `g(u) = a(u) \conv b(u*_ratio) = Integrate[a(u-t)*b(t*_ratio),dt]` and there's no issue with uniform scaling.
	// NOTE: Blit Utils want `f(x) = a(x/c_1)/c_1 \conv b(x/c_2)/c_2` where often `c_1 = 1`
	inline CConvolutionWeightFunction1D(WeightFunction1DA&& funcA, WeightFunction1DB&& funcB)
		: m_funcA(std::move(funcA)), m_funcB(std::move(funcB))
	{
		m_minSupport = m_funcA.getMinSupport() + m_funcB.getMinSupport();
		m_maxSupport = m_funcA.getMaxSupport() + m_funcB.getMaxSupport();
	}

	template<int32_t derivative>
	double operator()(const float x, const uint32_t channel, const uint32_t sampleCount = 64u) const
	{
		return impl::convolution_weight_function_helper<WeightFunction1DA, WeightFunction1DB>::operator_impl<derivative>(*this, x, channel, sampleCount);
	}

	inline float getMinSupport() const { m_minSupport; }
	inline float getMaxSupport() const { m_maxSupport; }

	// If we want to allow the user to stretch and scale the convolution function we have to implement those methods (stretch, scale and stretchAndScale) here separately
	// in terms of WeightFunction1DA and WeightFunction1DB's corresponding methods.

private:
	float m_minSupport;
	float m_maxSupport;
};

} // end namespace nbl::asset

#endif