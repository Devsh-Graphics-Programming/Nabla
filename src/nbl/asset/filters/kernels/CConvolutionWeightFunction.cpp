#include "nbl/asset/filters/kernels/CConvolutionWeightFunction.h"

namespace nbl::asset
{

template <>
double CConvolutionWeightFunction1D<CWeightFunction1D<SBoxFunction>, CWeightFunction1D<SBoxFunction>>::weight_impl(const float x, const uint32_t channel, const uint32_t) const
{
	const auto [minIntegrationLimit, maxIntegrationLimit] = getIntegrationDomain(x);

	if (m_isFuncAWider)
		return (maxIntegrationLimit - minIntegrationLimit) * m_funcA.weight(x, channel) * m_funcB.weight(0.f, channel);
	else
		return (maxIntegrationLimit - minIntegrationLimit) * m_funcB.weight(x, channel) * m_funcA.weight(0.f, channel);
}

template <>
double CConvolutionWeightFunction1D<CWeightFunction1D<SGaussianFunction>, CWeightFunction1D<SGaussianFunction>>::weight_impl(const float x, const uint32_t channel, const uint32_t) const
{
	const float funcA_stddev = 1.f / (m_funcA.weight(0.f, channel) * core::sqrt(2.f * core::PI<float>()));
	const float funcB_stddev = 1.f / (m_funcB.weight(0.f, channel) * core::sqrt(2.f * core::PI<float>()));

	const float convolution_stddev = core::sqrt(funcA_stddev * funcA_stddev + funcB_stddev * funcB_stddev);
	asset::CWeightFunction1D<asset::SGaussianFunction> weightFunction;
	weightFunction.stretchAndScale(convolution_stddev);

	return weightFunction.weight(x, channel);
}

template <>
double CConvolutionWeightFunction1D<CWeightFunction1D<SKaiserFunction>, CWeightFunction1D<SKaiserFunction>>::weight_impl(const float x, const uint32_t channel, const uint32_t) const
{
	return m_isFuncAWider ? m_funcA.weight(x, channel) : m_funcB.weight(x, channel);
}

} // end namespace nbl::asset