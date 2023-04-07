// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_CONVOLUTION_WEIGHT_FUNCTION_H_INCLUDED_
#define _NBL_ASSET_C_CONVOLUTION_WEIGHT_FUNCTION_H_INCLUDED_

#include "nbl/asset/filters/kernels/WeightFunctions.h"

namespace nbl::asset
{

// TODO(achal): I can also make a concept for WeightFunction1D here so that it only allows the type CWeightFunction1D.

// If we allow the user to specify a derivative of CConvolutionWeightFunction1D then there will be the follwing problem:
// There will be no way for us to evaluate the constituent functions (m_funcA, m_funcB) on an arbitrary derivative because
// `derivative` is a class template member not a function template member (done to handle chain rule) of CWeightFunction1D.
// So, we would have to create new `CWeightFunction1D`s entirely and use them (not just here, but in the various specializations
// of this function). Not sure if its worth it, both in terms of performance and code complexity.

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

	inline void stretch(const float s)
	{
		assert(s != 0.f);

		m_minSupport *= s;
		m_maxSupport *= s;

		const float rcp_s = 1.f / s;

		m_invStretch *= rcp_s;
	}

	inline void scale(const float s)
	{
		assert(s != 0.f);
        m_totalScale *= s;
    }

	inline void stretchAndScale(const float stretchFactor)
	{
		stretch(stretchFactor);
		scale(1.f / stretchFactor);
	}

	inline double weight(const float x, const uint8_t channel, const uint32_t sampleCount = 64u) const
	{
		return static_cast<double>(m_totalScale * weight_impl(x * m_invStretch, channel, sampleCount));
	}

	inline float getMinSupport() const { return m_minSupport; }
	inline float getMaxSupport() const { return m_maxSupport; }

private:
	const WeightFunction1DA m_funcA;
	const WeightFunction1DB m_funcB;

	const bool m_isFuncAWider;
	float m_minSupport;
	float m_maxSupport;
	float m_invStretch = 1.f;
	float m_totalScale = 1.f;

	double weight_impl(const float x, const uint32_t channel, const uint32_t sampleCount) const
	{
		if constexpr (std::is_same_v<WeightFunction1DB, CWeightFunction1D<SDiracFunction>>)
		{
			return m_funcA.weight(x, channel);
		}
		else if (std::is_same_v<WeightFunction1DA, CWeightFunction1D<SDiracFunction>>)
		{
			return m_funcB.weight(x, channel);
		}
		else
		{
			auto [minIntegrationLimit, maxIntegrationLimit] = getIntegrationDomain(x);

			// TODO(achal): what ????
			// if this happens, it means that `m_ratio=INF` and it degenerated into a dirac delta
			if (minIntegrationLimit == maxIntegrationLimit)
			{
				assert(false);
				assert(WeightFunction1DB::k_energy[channel] != 0.f); // TODO(achal): Remove.
				return m_funcA.weight(x, channel) * WeightFunction1DB::k_energy[channel];
			}

			const double dtau = (maxIntegrationLimit - minIntegrationLimit) / sampleCount;

			// TODO(achal): what ???
			// if this happened then `m_ratio=0` and we have infinite domain, this is not a problem
			if (core::isnan<double>(dtau))
			{
				assert(false);
				return m_funcA.weight(x, channel) * m_funcB.weight(0.f, channel);
			}

			double result = 0.0;
			for (uint32_t i = 0u; i < sampleCount; ++i)
			{
				const double tau = minIntegrationLimit + i * dtau;
				if (m_isFuncAWider)
					result += m_funcA.weight(tau, channel) * m_funcB.weight(x - tau, channel) * dtau;
				else
					result += m_funcB.weight(tau, channel) * m_funcA.weight(x - tau, channel) * dtau;
			}
			return static_cast<float>(result);
		}
	}

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

template <>
double CConvolutionWeightFunction1D<CWeightFunction1D<SBoxFunction>, CWeightFunction1D<SBoxFunction>>::weight_impl(const float x, const uint32_t channel, const uint32_t) const;

template <>
double CConvolutionWeightFunction1D<CWeightFunction1D<SGaussianFunction>, CWeightFunction1D<SGaussianFunction>>::weight_impl(const float x, const uint32_t channel, const uint32_t) const;

// TODO: Specialization: CConvolutionWeightFunction1D<CWeightFunction1D<STriangleFunction>, CWeightFunction1D<STriangleFunction>> = this is tricky but feasible

template <>
double CConvolutionWeightFunction1D<CWeightFunction1D<SKaiserFunction>, CWeightFunction1D<SKaiserFunction>>::weight_impl(const float x, const uint32_t channel, const uint32_t) const;

} // end namespace nbl::asset

#endif