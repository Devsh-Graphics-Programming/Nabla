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
	inline static float operator_impl(const CConvolutionWeightFunction1D<WeightFunction1DA, WeightFunction1DB>& _this, const float x, const uint32_t channel, const uint32_t sampleCount)
	{
		if constexpr (std::is_same_v<WeightFunction1DB, CWeightFunction1D<SDiracFunction>>)
		{
			return _this.m_funcA.operator()(x, channel);
		}
		else if (std::is_same_v<WeightFunction1DA, CWeightFunction1D<SDiracFunction>>)
		{
			return _this.m_funcB.operator()(x, channel);
		}
		else
		{
			// constexpr auto deriv_A = std::min(static_cast<int32_t>(WeightFunction1DA::k_smoothness), derivative);
			// constexpr auto deriv_B = derivative - deriv_A;

			auto [minIntegrationLimit, maxIntegrationLimit] = _this.getIntegrationDomain(x);

			// TODO(achal): what ????
			// if this happens, it means that `m_ratio=INF` and it degenerated into a dirac delta
			if (minIntegrationLimit == maxIntegrationLimit)
			{
				assert(false);
				assert(WeightFunction1DB::k_energy[channel] != 0.f); // TODO(achal): Remove.
				return _this.m_funcA.operator()(x, channel) * WeightFunction1DB::k_energy[channel];
			}

			const double dtau = (maxIntegrationLimit - minIntegrationLimit) / sampleCount;
			// TODO(achal): what ???
			// if this happened then `m_ratio=0` and we have infinite domain, this is not a problem
			// if (core::isnan<double>(dt))
			// 	return _this.m_funcA.operator()(x, channel) * _this.m_funcB.operator()(0.f, channel);

			double result = 0.0;
			for (uint32_t i = 0u; i < sampleCount; ++i)
			{
				const double tau = minIntegrationLimit + i * dtau;
				if (_this.m_isFuncAWider)
					result += _this.m_funcA.operator()(tau, channel) * _this.m_funcB.operator()(x-tau, channel) * dtau;
				else
					result += _this.m_funcB.operator()(tau, channel) * _this.m_funcA.operator()(x-tau, channel) * dtau;
			}
			return static_cast<float>(result);
		}
	}
};

template <>
struct convolution_weight_function_helper<CWeightFunction1D<SBoxFunction>, CWeightFunction1D<SBoxFunction>>
{
	static float operator_impl(const CConvolutionWeightFunction1D<CWeightFunction1D<SBoxFunction>, CWeightFunction1D<SBoxFunction>>& _this, const float x, const uint32_t channel, const uint32_t sampleCount);
};

template <>
struct convolution_weight_function_helper<CWeightFunction1D<SGaussianFunction>, CWeightFunction1D<SGaussianFunction>>
{
	static float operator_impl(const CConvolutionWeightFunction1D<CWeightFunction1D<SGaussianFunction>, CWeightFunction1D<SGaussianFunction>>& _this, const float x, const uint32_t channel, const uint32_t sampleCount);
};

// TODO: Specialization: convolution_weight_function_helper<Triangle,Triangle> = this is tricky but feasible

template <>
struct convolution_weight_function_helper<CWeightFunction1D<SKaiserFunction>, CWeightFunction1D<SKaiserFunction>>
{
	static float operator_impl(const CConvolutionWeightFunction1D<CWeightFunction1D<SKaiserFunction>, CWeightFunction1D<SKaiserFunction>>& _this, const float x, const uint32_t channel, const uint32_t sampleCount);
};

} // end namespace impl

// TODO(achal): I can also make a concept for WeightFunction1D here so that it only allows the type CWeightFunction1D.

// this is the horribly slow generic version that you should not use (only use the specializations or when one of the weights is a dirac)
template<typename WeightFunction1DA, typename WeightFunction1DB>
class CConvolutionWeightFunction1D
{
	// TODO(achal): Not passing.
	// static_assert(std::is_same_v<impl::weight_function_value_type_t<WeightFunction1DA>, impl::weight_function_value_type_t<WeightFunction1DB>>, "Both functions must use the same Value Type!");

public:
	constexpr static inline uint32_t k_smoothness = WeightFunction1DA::k_smoothness + WeightFunction1DB::k_smoothness;

	inline CConvolutionWeightFunction1D(WeightFunction1DA&& funcA, WeightFunction1DB&& funcB)
		: m_funcA(std::move(funcA)), m_funcB(std::move(funcB)),
		m_isFuncAWider((m_funcA.getMaxSupport() - m_funcA.getMinSupport()) > (m_funcB.getMaxSupport() - m_funcB.getMinSupport()))
	{
		m_minSupport = m_funcA.getMinSupport() + m_funcB.getMinSupport();
		m_maxSupport = m_funcA.getMaxSupport() + m_funcB.getMaxSupport();
	}

	// TODO(achal): I we want to allow taking derivative of CConvolutionWeightFunction1D then this template param have to go to the class i.e. CConvolutionWeightFunction1D
	// and then the chain rule should be handled in the (not yet implemented) CConvolutionWeightFunction1D::stretch --if we want to allow stretching that is,
	// much like CWeightFunction1D does.
	// template<int32_t derivative>
	double operator()(const float x, const uint32_t channel, const uint32_t sampleCount = 64u) const
	{
		return impl::convolution_weight_function_helper<WeightFunction1DA, WeightFunction1DB>::operator_impl(*this, x, channel, sampleCount);
	}

	inline float getMinSupport() const { return m_minSupport; }
	inline float getMaxSupport() const { return m_maxSupport; }

	// If we want to allow the user to stretch and scale the convolution function we have to implement those methods (stretch, scale and stretchAndScale) here separately
	// in terms of WeightFunction1DA and WeightFunction1DB's corresponding methods.

private:
	friend struct impl::convolution_weight_function_helper<WeightFunction1DA, WeightFunction1DB>;
	const WeightFunction1DA m_funcA;
	const WeightFunction1DB m_funcB;

	const bool m_isFuncAWider;
	float m_minSupport;
	float m_maxSupport;

	std::pair<double, double> getIntegrationDomain(const float x) const
	{
		// The following if-else checks to figure out integration domain assumes that the wider function
		// is stationary while the narrow one is "moving".
		// We assume that the wider kernel is stationary (not shifting as `x` changes) while the narrower kernel is the one which shifts, such that it is always centered at x.

		const float funcNarrowMinSupport = m_isFuncAWider ? m_funcB.getMinSupport() : m_funcA.getMinSupport();
		const float funcNarrowMaxSupport = m_isFuncAWider ? m_funcB.getMaxSupport() : m_funcA.getMaxSupport();

		const float funcWideMinSupport = m_isFuncAWider ? m_funcA.getMinSupport() : m_funcB.getMinSupport();
		const float funcWideMaxSupport = m_isFuncAWider ? m_funcA.getMaxSupport() : m_funcB.getMaxSupport();

		const float funcNarrowWidth = funcNarrowMaxSupport - funcNarrowMinSupport;
		const float funcWideWidth = funcWideMaxSupport - funcWideMinSupport;

		const float funcNarrowWidth_half = funcNarrowWidth * 0.5;

		double minIntegrationLimit = 0.0, maxIntegrationLimit = 0.0;
		{
			if ((x >= (funcWideMinSupport - funcNarrowWidth_half)) && (x <= (funcWideMinSupport + funcNarrowWidth_half)))
			{
				minIntegrationLimit = funcWideMinSupport;
				maxIntegrationLimit = x + funcNarrowWidth_half;
			}
			else if ((x >= (funcWideMinSupport + funcNarrowWidth_half)) && (x <= (funcWideMaxSupport - funcNarrowWidth_half)))
			{
				minIntegrationLimit = x - funcNarrowWidth_half;
				maxIntegrationLimit = x + funcNarrowWidth_half;
			}
			else if ((x >= (funcWideMaxSupport - funcNarrowWidth_half)) && (x <= (funcWideMaxSupport + funcNarrowWidth_half)))
			{
				minIntegrationLimit = x - funcNarrowWidth_half;
				maxIntegrationLimit = funcWideMaxSupport;
			}
		}
		assert(minIntegrationLimit <= maxIntegrationLimit);

		return { minIntegrationLimit, maxIntegrationLimit };
	}
};

} // end namespace nbl::asset

#endif