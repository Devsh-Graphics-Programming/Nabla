#include "nbl/asset/filters/kernels/CConvolutionImageFilterKernel.h"

namespace nbl::asset
{

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CBoxImageFilterKernel>, CScaledImageFilterKernel<CBoxImageFilterKernel>>::weight(const float x, const uint32_t channel, const uint32_t unused) const
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

// Dirac with Box

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CDiracImageFilterKernel>, CScaledImageFilterKernel<CBoxImageFilterKernel>>::weight(const float x, const uint32_t channel, const uint32_t) const
{
	return m_kernelB.weight(x, channel);
}

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CBoxImageFilterKernel>, CScaledImageFilterKernel<CDiracImageFilterKernel>>::weight(const float x, const uint32_t channel, const uint32_t) const
{
	return m_kernelA.weight(x, channel);
}

// Dirac with Triangle

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CDiracImageFilterKernel>, CScaledImageFilterKernel<CTriangleImageFilterKernel>>::weight(const float x, const uint32_t channel, const uint32_t) const
{
	return m_kernelB.weight(x, channel);
}

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CTriangleImageFilterKernel>, CScaledImageFilterKernel<CDiracImageFilterKernel>>::weight(const float x, const uint32_t channel, const uint32_t) const
{
	return m_kernelA.weight(x, channel);
}

// Dirac with Gaussian

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CDiracImageFilterKernel>, CScaledImageFilterKernel<CGaussianImageFilterKernel<>>>::weight(const float x, const uint32_t channel, const uint32_t) const
{
	return m_kernelB.weight(x, channel);
}

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CGaussianImageFilterKernel<>>, CScaledImageFilterKernel<CDiracImageFilterKernel>>::weight(const float x, const uint32_t channel, const uint32_t) const
{
	return m_kernelA.weight(x, channel);
}

// Dirac with Mitchell

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CDiracImageFilterKernel>, CScaledImageFilterKernel<CMitchellImageFilterKernel<>>>::weight(const float x, const uint32_t channel, const uint32_t) const
{
	return m_kernelB.weight(x, channel);
}

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CMitchellImageFilterKernel<>>, CScaledImageFilterKernel<CDiracImageFilterKernel>>::weight(const float x, const uint32_t channel, const uint32_t) const
{
	return m_kernelA.weight(x, channel);
}

// Dirac with Kaiser

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CDiracImageFilterKernel>, CScaledImageFilterKernel<CKaiserImageFilterKernel<>>>::weight(const float x, const uint32_t channel, const uint32_t) const
{
	return m_kernelB.weight(x, channel);
}

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CKaiserImageFilterKernel<>>, CScaledImageFilterKernel<CDiracImageFilterKernel>>::weight(const float x, const uint32_t channel, const uint32_t) const
{
	return m_kernelA.weight(x, channel);
}

}