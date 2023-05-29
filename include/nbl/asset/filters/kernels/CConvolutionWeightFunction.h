// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_CONVOLUTION_WEIGHT_FUNCTION_H_INCLUDED_
#define _NBL_ASSET_C_CONVOLUTION_WEIGHT_FUNCTION_H_INCLUDED_

#include "nbl/asset/filters/kernels/WeightFunctions.h"

namespace nbl::asset
{

// We don't allow the user to specify a derivative of CConvolutionWeightFunction1D as a template parameter on the `weight` function
// because there would be no way for us to evaluate the constituent functions (m_funcA, m_funcB) on an arbitrary derivative because
// `derivative` is a class template member not a function template member (done to handle chain rule) of CWeightFunction1D.
// So, we would have to create new `CWeightFunction1D`s entirely and use them in the various specializations of the `weight_impl`.

// If you want a Nth order derivative of `Conv<CWeightFunction<A>,CWeightFunction<B>>` then create a `Conv<CWeightFunction<Ai,>,CWeightFunction<B,j>>`
// which satisfies `i<A::k_smoothness && j<B::k_smoothness && (i+j)==n` which is basically applying the derivative property of convolution.

// This is the horribly slow generic version that you should not use (only use the specializations or when one of the weights is a dirac)
template<SimpleWeightFunction1D WeightFunction1DA, SimpleWeightFunction1D WeightFunction1DB>
class CConvolutionWeightFunction1D final : public impl::IWeightFunction1D<WeightFunction1DA::value_t>
{
		static_assert(std::is_same_v<WeightFunction1DA::value_t,WeightFunction1DB::value_t>, "Both functions must use the same Value Type!");
	public:
		constexpr static inline uint32_t k_smoothness = WeightFunction1DA::k_smoothness + WeightFunction1DB::k_smoothness;

		inline CConvolutionWeightFunction1D(WeightFunction1DA&& funcA, WeightFunction1DB&& funcB)
			: impl::IWeightFunction1D<WeightFunction1DA::value_t>(
				m_funcA.getMinSupport()+m_funcB.getMinSupport(),
				m_funcA.getMaxSupport()+m_funcB.getMaxSupport()
			), m_funcA(std::move(funcA)), m_funcB(std::move(funcB))
		{
		}

		inline void stretch(const float s) {impl_stretch(s);}

		// This method will keep the integral of the weight function without derivatives constant.
		inline void stretchAndScale(const float stretchFactor)
		{
			stretch(stretchFactor);
			scale(value_t(1)/stretchFactor);
		}

		inline value_t weight(float x, const uint32_t sampleCount = 64u) const
		{
			value_t retval;
			// handle global stretch before we do anything
			x *= m_invStretch;
			if constexpr (std::is_same_v<WeightFunction1DB::function_t,SDiracFunction> && WeightFunction1DB::k_derivative==0)
				retval = m_funcA.weight(x);
			else if (std::is_same_v<WeightFunction1DA::function_t,SDiracFunction> && WeightFunction1DA::k_derivative==0)
				retval = m_funcB.weight(x);
			else
			{
				// Handle generate cases of constituent function degenerate scaling: https://github.com/Devsh-Graphics-Programming/Nabla/pull/435/files#r1206984286
				// A 
				if (core::is_inf(m_funcA.m_invStretch))
					retval = m_funcA.energy();
				else if (core::abs(m_funcA.m_invStretch)<=std::numeric_limits<value_t>::min())
					retval = m_funcA.weight(0.f);
				else
					retval = 0.0/0.0; // abuse NaN to indicate `A.m_invStretch` is finite
				// B
				if (core::is_inf(m_funcB.m_invStretch))
				{
					if (core::isnan(retval))
						retval = m_funcA.weight(x);
					retval *= m_funcB.energy();
				}
				else if (core::abs(m_funcB.m_invStretch)<=std::numeric_limits<value_t>::min())
				{
					if (core::isnan(retval))
						retval = m_funcA.weight(x);
					retval *= m_funcB.weight(0.f);
				}
				else if (core::isnan(retval)) // NaN means scale was finite
					retval = weight_impl(x, sampleCount);
				else
					retval *= m_funcB.weight(x);
			}
			return m_totalScale * retval;
		}

	private:
		const WeightFunction1DA m_funcA;
		const WeightFunction1DB m_funcB;

		value_t weight_impl(const float x, const uint32_t sampleCount) const
		{
			// simple AABB intersection, note that 
			const double minIntegrationLimit = std::max(m_funcA.getMinSupport(),x-m_funcB.getMaxSupport());
			const double maxIntegrationLimit = std::min(m_funcB.getMaxSupport(),x-m_funcB.getMinSupport());
			const double dtau = (maxIntegrationLimit-minIntegrationLimit)/double(sampleCount);
			// only proceed with integration if integration domain is valid (non zero size and correct orientation)
			value_t result = 0.0;
			// otherwise no overlap between the two functions when shifted by x
			if (dtau>std::numeric_limits<value_t>::min())
			for (uint32_t i=0u; i<sampleCount; ++i)
			{
				const double tau = dtau*double(i)+minIntegrationLimit;
				result += m_funcA.weight(tau, channel) * m_funcB.weight(x-tau, channel) * dtau;
			}
			return static_cast<value_t>(result);
		}
};

template <>
double CConvolutionWeightFunction1D<CWeightFunction1D<SBoxFunction>, CWeightFunction1D<SBoxFunction>>::weight_impl(const float x, const uint32_t) const;

template <>
double CConvolutionWeightFunction1D<CWeightFunction1D<SGaussianFunction>, CWeightFunction1D<SGaussianFunction>>::weight_impl(const float x, const uint32_t) const;

// TODO: Specialization: CConvolutionWeightFunction1D<CWeightFunction1D<STriangleFunction>, CWeightFunction1D<STriangleFunction>> = this is tricky but feasible

template <>
double CConvolutionWeightFunction1D<CWeightFunction1D<SKaiserFunction>, CWeightFunction1D<SKaiserFunction>>::weight_impl(const float x, const uint32_t) const;

} // end namespace nbl::asset

#endif
