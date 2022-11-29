#include "nbl/asset/filters/kernels/CConvolutionImageFilterKernel.h"

namespace nbl::asset
{

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CBoxImageFilterKernel>, CScaledImageFilterKernel<CBoxImageFilterKernel>>::weight(const float x, const uint32_t channel, const uint32_t unused) const
{
	const auto [minIntegrationLimit, maxIntegrationLimit, domainType] = getIntegrationDomain(x);
	return (maxIntegrationLimit - minIntegrationLimit) / (getKernelWidth(m_kernelA)*getKernelWidth(m_kernelB));
}

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CKaiserImageFilterKernel<>>, CScaledImageFilterKernel<CKaiserImageFilterKernel<>>>::weight(const float x, const uint32_t channel, const uint32_t) const
{
	return getKernelWidth(m_kernelA) > getKernelWidth(m_kernelB) ? m_kernelA.weight(x, channel) : m_kernelB.weight(x, channel);
}

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CGaussianImageFilterKernel<>>, CScaledImageFilterKernel<CGaussianImageFilterKernel<>>>::weight(const float x, const uint32_t channel, const uint32_t) const
{
	const float kernelA_stddev = IImageFilterKernel::ScaleFactorUserData::cast(m_kernelA.getUserData())->factor[channel];
	const float kernelB_stddev = IImageFilterKernel::ScaleFactorUserData::cast(m_kernelB.getUserData())->factor[channel];
	const float convolution_stddev = core::sqrt(kernelA_stddev * kernelA_stddev + kernelB_stddev * kernelB_stddev);

	auto convolutionKernel = asset::CScaledImageFilterKernel<CGaussianImageFilterKernel<>>(core::vectorSIMDf(convolution_stddev, 0.f, 0.f, 0.f), asset::CGaussianImageFilterKernel<>());

	return convolutionKernel.weight(x, channel);
}

}