#include "nbl/asset/filters/kernels/CConvolutionWeightFunction.h"

namespace nbl::asset::impl
{

float convolution_weight_function_helper<CWeightFunction1D<SBoxFunction>, CWeightFunction1D<SBoxFunction>>::operator_impl(const CConvolutionWeightFunction1D<CWeightFunction1D<SBoxFunction>, CWeightFunction1D<SBoxFunction>>& _this, const float x, const uint32_t channel, const uint32_t sampleCount)
{
	const auto [minIntegrationLimit, maxIntegrationLimit] = _this.getIntegrationDomain(x);

	if (_this.m_isFuncAWider)
		return (maxIntegrationLimit - minIntegrationLimit) * _this.m_funcA(x, channel) * _this.m_funcB(0.f, channel);
	else
		return (maxIntegrationLimit - minIntegrationLimit) * _this.m_funcB(x, channel) * _this.m_funcA(0.f, channel);
}

float convolution_weight_function_helper<CWeightFunction1D<SGaussianFunction>, CWeightFunction1D<SGaussianFunction>>::operator_impl(const CConvolutionWeightFunction1D<CWeightFunction1D<SGaussianFunction>, CWeightFunction1D<SGaussianFunction>>& _this, const float x, const uint32_t channel, const uint32_t sampleCount)
{
	const float funcA_stddev = 1.f/(_this.m_funcA(0.f, channel)*core::sqrt(2.f*core::PI<float>()));
	const float funcB_stddev = 1.f/(_this.m_funcB(0.f, channel)*core::sqrt(2.f*core::PI<float>()));

	const float convolution_stddev = core::sqrt(funcA_stddev * funcA_stddev + funcB_stddev * funcB_stddev);
	asset::CWeightFunction1D<asset::SGaussianFunction> weightFunction;
	weightFunction.stretchAndScale(convolution_stddev);

	return weightFunction(x, channel);
}

float convolution_weight_function_helper<CWeightFunction1D<SKaiserFunction>, CWeightFunction1D<SKaiserFunction>>::operator_impl(const CConvolutionWeightFunction1D<CWeightFunction1D<SKaiserFunction>, CWeightFunction1D<SKaiserFunction>>& _this, const float x, const uint32_t channel, const uint32_t sampleCount)
{
	return _this.m_isFuncAWider ? _this.m_funcA(x, channel) : _this.m_funcB(x, channel);
}

} // end namespace nbl::asset