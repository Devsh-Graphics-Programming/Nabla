#include "nbl/asset/filters/kernels/CConvolutionWeightFunction.h"

namespace nbl::asset
{

template <>
double CConvolutionWeightFunction1D<CWeightFunction1D<SBoxFunction>, CWeightFunction1D<SBoxFunction>>::weight_impl(const float x, const uint32_t) const
{
	const auto [minIntegrationLimit,maxIntegrationLimit] = getIntegrationDomain(x);
	if (minIntegrationLimit<maxIntegrationLimit)
		return (maxIntegrationLimit-minIntegrationLimit)*m_funcA.getTotalScale()*m_funcB.getTotalScale();
	// else no overlap
	return 0.0;
}

template <>
double CConvolutionWeightFunction1D<CWeightFunction1D<SGaussianFunction<>>, CWeightFunction1D<SGaussianFunction<>>>::weight_impl(const float x, const uint32_t) const
{
	asset::CWeightFunction1D<asset::SGaussianFunction<>> weightFunction;

	const float funcA_stddev = 1.f/m_funcA.getInvStretch();
	const float funcB_stddev = 1.f/m_funcB.getInvStretch();
	const float convolution_stddev = core::sqrt(funcA_stddev * funcA_stddev + funcB_stddev * funcB_stddev);
	weightFunction.stretchAndScale(convolution_stddev);
	
	const double funcA_true_scale = m_funcA.getTotalScale()*funcA_stddev;
	const double funcB_true_scale = m_funcB.getTotalScale()*funcB_stddev;
	weightFunction.scale(funcA_true_scale*funcB_true_scale);

	return weightFunction.weight(x);
}

template <>
double CConvolutionWeightFunction1D<CWeightFunction1D<SKaiserFunction>, CWeightFunction1D<SKaiserFunction>>::weight_impl(const float x, const uint32_t) const
{
	const double true_scale = m_funcA.getTotalScale()*m_funcB.getTotalScale()/(m_funcA.getInvStretch()*m_funcB.getInvStretch());
	const double support_width_a = m_funcA.getMaxSupport() - m_funcA.getMinSupport();
	const double support_width_b = m_funcB.getMaxSupport() - m_funcB.getMinSupport();
	if (support_width_a>support_width_b)
		return true_scale*m_funcA.weight(x);
	else
		return true_scale*m_funcB.weight(x);
}

} // end namespace nbl::asset
