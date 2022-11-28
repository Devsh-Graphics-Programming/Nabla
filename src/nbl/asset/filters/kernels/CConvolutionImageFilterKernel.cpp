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

}