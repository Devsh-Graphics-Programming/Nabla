#include "nbl/asset/filters/kernels/CConvolutionImageFilterKernel.h"

namespace nbl::asset
{

template <>
float CConvolutionImageFilterKernel<CBoxImageFilterKernel, CBoxImageFilterKernel>::weight(const float x, const uint32_t channel, const uint32_t unused) const
{
	const auto [minIntegrationLimit, maxIntegrationLimit] = getIntegrationDomain(x);

	const float kernelAWidth = getKernelWidth(m_kernelA);
	const float kernelBWidth = getKernelWidth(m_kernelB);

	const auto& kernelNarrow = kernelAWidth < kernelBWidth ? m_kernelA : m_kernelB;
	const auto& kernelWide = kernelAWidth > kernelBWidth ? m_kernelA : m_kernelB;

	// We assume that the wider kernel is stationary (not shifting as `x` changes) while the narrower kernel is the one which shifts, such that it is always centered at x.
	return (maxIntegrationLimit - minIntegrationLimit) * kernelWide.weight(x, channel) * kernelNarrow.weight(0.f, channel);
}

template <>
float CConvolutionImageFilterKernel<CKaiserImageFilterKernel<>, CKaiserImageFilterKernel<>>::weight(const float x, const uint32_t channel, const uint32_t) const
{
	return getKernelWidth(m_kernelA) > getKernelWidth(m_kernelB) ? m_kernelA.weight(x, channel) : m_kernelB.weight(x, channel);
}

template <>
float CConvolutionImageFilterKernel<CGaussianImageFilterKernel<>, CGaussianImageFilterKernel<>>::weight(const float x, const uint32_t channel, const uint32_t) const
{
	const float kernelA_stddev = m_kernelA.m_multipliedScale[channel];
	const float kernelB_stddev = m_kernelB.m_multipliedScale[channel];
	const float convolution_stddev = core::sqrt(kernelA_stddev * kernelA_stddev + kernelB_stddev * kernelB_stddev);

	const auto stretchFactor = core::vectorSIMDf(convolution_stddev, 1.f, 1.f, 1.f);
	auto convolutionKernel = asset::CGaussianImageFilterKernel<>();
	convolutionKernel.stretchAndScale(stretchFactor);

	return convolutionKernel.weight(x, channel);
}

// Dirac with Box

template <>
float CConvolutionImageFilterKernel<CDiracImageFilterKernel, CBoxImageFilterKernel>::weight(const float x, const uint32_t channel, const uint32_t) const
{
	return m_kernelB.weight(x, channel);
}

template <>
float CConvolutionImageFilterKernel<CBoxImageFilterKernel, CDiracImageFilterKernel>::weight(const float x, const uint32_t channel, const uint32_t) const
{
	return m_kernelA.weight(x, channel);
}

// Dirac with Triangle

template <>
float CConvolutionImageFilterKernel<CDiracImageFilterKernel, CTriangleImageFilterKernel>::weight(const float x, const uint32_t channel, const uint32_t) const
{
	return m_kernelB.weight(x, channel);
}

template <>
float CConvolutionImageFilterKernel<CTriangleImageFilterKernel, CDiracImageFilterKernel>::weight(const float x, const uint32_t channel, const uint32_t) const
{
	return m_kernelA.weight(x, channel);
}

// Dirac with Gaussian

template <>
float CConvolutionImageFilterKernel<CDiracImageFilterKernel, CGaussianImageFilterKernel<>>::weight(const float x, const uint32_t channel, const uint32_t) const
{
	return m_kernelB.weight(x, channel);
}

template <>
float CConvolutionImageFilterKernel<CGaussianImageFilterKernel<>, CDiracImageFilterKernel>::weight(const float x, const uint32_t channel, const uint32_t) const
{
	return m_kernelA.weight(x, channel);
}

// Dirac with Mitchell

template <>
float CConvolutionImageFilterKernel<CDiracImageFilterKernel, CMitchellImageFilterKernel<>>::weight(const float x, const uint32_t channel, const uint32_t) const
{
	return m_kernelB.weight(x, channel);
}

template <>
float CConvolutionImageFilterKernel<CMitchellImageFilterKernel<>, CDiracImageFilterKernel>::weight(const float x, const uint32_t channel, const uint32_t) const
{
	return m_kernelA.weight(x, channel);
}

// Dirac with Kaiser

template <>
float CConvolutionImageFilterKernel<CDiracImageFilterKernel, CKaiserImageFilterKernel<>>::weight(const float x, const uint32_t channel, const uint32_t) const
{
	return m_kernelB.weight(x, channel);
}

template <>
float CConvolutionImageFilterKernel<CKaiserImageFilterKernel<>, CDiracImageFilterKernel>::weight(const float x, const uint32_t channel, const uint32_t) const
{
	return m_kernelA.weight(x, channel);
}

}