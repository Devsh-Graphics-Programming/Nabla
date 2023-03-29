#include "nbl/asset/filters/kernels/CConvolutionWeightFunction.h"

namespace nbl::asset::impl
{

template <int32_t derivative>
float convolution_weight_function_helper<SBoxFunction, SBoxFunction>::operator_impl(const CConvolutionWeightFunction<SBoxFunction, SBoxFunction>& _this, const float x, const uint32_t channel, const uint32_t sampleCount)
{
	assert(false);

	const auto [minIntegrationLimit, maxIntegrationLimit] = getIntegrationDomain(x);

	const float kernelAWidth = getKernelWidth(m_kernelA);
	const float kernelBWidth = getKernelWidth(m_kernelB);

	const auto& kernelNarrow = kernelAWidth < kernelBWidth ? m_kernelA : m_kernelB;
	const auto& kernelWide = kernelAWidth > kernelBWidth ? m_kernelA : m_kernelB;

	// We assume that the wider kernel is stationary (not shifting as `x` changes) while the narrower kernel is the one which shifts, such that it is always centered at x.
	return (maxIntegrationLimit - minIntegrationLimit) * kernelWide.weight(x, channel) * kernelNarrow.weight(0.f, channel);
}

template <int32_t derivative>
float convolution_weight_function_helper<SGaussianFunction, SGaussianFunction>::operator_impl(const CConvolutionWeightFunction<SGaussianFunction, SGaussianFunction>& _this, const float x, const uint32_t channel, const uint32_t sampleCount)
{
	assert(false);

	const float kernelA_stddev = m_kernelA.m_multipliedScale[channel];
	const float kernelB_stddev = m_kernelB.m_multipliedScale[channel];
	const float convolution_stddev = core::sqrt(kernelA_stddev * kernelA_stddev + kernelB_stddev * kernelB_stddev);

	const auto stretchFactor = core::vectorSIMDf(convolution_stddev, 1.f, 1.f, 1.f);
	auto convolutionKernel = asset::CGaussianImageFilterKernel();
	convolutionKernel.stretchAndScale(stretchFactor);

	return convolutionKernel(x, channel);
}

template <int32_t derivative>
float convolution_weight_function_helper<SKaiserFunction, SKaiserFunction>::operator_impl(const CConvolutionWeightFunction<SKaiserFunction, SKaiserFunction>& _this, const float x, const uint32_t channel, const uint32_t sampleCount)
{
	return getKernelWidth(m_kernelA) > getKernelWidth(m_kernelB) ? m_kernelA.weight(x, channel) : m_kernelB.weight(x, channel);
}

} // end namespace nbl::asset