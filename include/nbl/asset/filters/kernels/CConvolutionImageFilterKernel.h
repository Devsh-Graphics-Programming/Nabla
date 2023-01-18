// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_CONVOLUTION_IMAGE_FILTER_KERNEL_H_INCLUDED__
#define __NBL_ASSET_C_CONVOLUTION_IMAGE_FILTER_KERNEL_H_INCLUDED__

#include "nbl/asset/filters/kernels/CBoxImageFilterKernel.h"
#include "nbl/asset/filters/kernels/CTriangleImageFilterKernel.h"
#include "nbl/asset/filters/kernels/CGaussianImageFilterKernel.h"
#include "nbl/asset/filters/kernels/CKaiserImageFilterKernel.h"
#include "nbl/asset/filters/kernels/CMitchellImageFilterKernel.h"
#include "nbl/asset/filters/kernels/CDiracImageFilterKernel.h"

namespace nbl::asset
{

// this is the horribly slow generic version that you should not use (only use the specializations)
template<typename KernelA, typename KernelB>
class NBL_API CConvolutionImageFilterKernel : public CFloatingPointSeparableImageFilterKernelBase<CConvolutionImageFilterKernel<KernelA, KernelB>>
{
	using Base = CFloatingPointSeparableImageFilterKernelBase<CConvolutionImageFilterKernel<KernelA, KernelB>>;

	static_assert(std::is_same_v<KernelA::value_type, KernelB::value_type>, "Both kernels must use the same value_type!");
	static_assert(KernelA::is_separable && KernelB::is_separable, "Convolving Non-Separable Filters is a TODO!");

public:
	CConvolutionImageFilterKernel(KernelA&& kernelA, KernelB&& kernelB)
		: Base(kernelA.negative_support.x + kernelB.negative_support.x, kernelA.positive_support.x + kernelB.positive_support.x),
		m_kernelA(std::move(kernelA)), m_kernelB(std::move(kernelB))
	{}

	float weight(const float x, const uint32_t channel, const uint32_t sampleCount = 64u) const
	{
		auto [minIntegrationLimit, maxIntegrationLimit] = getIntegrationDomain(x);

		if (minIntegrationLimit == maxIntegrationLimit)
			return 0.f;

		const double dt = (maxIntegrationLimit-minIntegrationLimit)/sampleCount;
		double result = 0.0;
		for (uint32_t i = 0u; i < sampleCount; ++i)
		{
			const double t = minIntegrationLimit + i*dt;
			result += m_kernelA.weight(t, channel) * m_kernelB.weight(x - t, channel) * dt;
		}

		return static_cast<float>(result);
	}

	static inline bool validate(ICPUImage* inImage, ICPUImage* outImage)
	{
		return KernelA::validate(inImage, outImage) && KernelB::validate(inImage, outImage);
	}

private:
	std::pair<double, double> getIntegrationDomain(const float x) const
	{
		// The following if-else checks to figure out integration domain assumes that the wider kernel
		// is stationary while the narrow one is "moving".
		// 
		// Also take this opportunity to add negative signs back into the negative_support.

		float kernelNarrowMinSupport = -m_kernelB.negative_support.x;
		float kernelNarrowMaxSupport = m_kernelB.positive_support.x;

		float kernelWideMinSupport = -m_kernelA.negative_support.x;
		float kernelWideMaxSupport = m_kernelA.positive_support.x;

		const float kernelAWidth = getKernelWidth(m_kernelA);
		const float kernelBWidth = getKernelWidth(m_kernelB);

		if (kernelAWidth < kernelBWidth)
		{
			std::swap(kernelNarrowMinSupport, kernelWideMinSupport);
			std::swap(kernelNarrowMaxSupport, kernelWideMaxSupport);
		}

		double minIntegrationLimit = 0.0, maxIntegrationLimit = 0.0;
		{
			if ((x > (kernelWideMinSupport + kernelNarrowMinSupport)) && (x <= (kernelWideMinSupport + kernelNarrowMaxSupport)))
			{
				minIntegrationLimit = -m_kernelA.negative_support.x;
				maxIntegrationLimit = x - (-m_kernelB.negative_support.x);
			}
			else if ((x > (kernelWideMinSupport + kernelNarrowMaxSupport)) && (x <= (kernelWideMaxSupport + kernelNarrowMinSupport)))
			{
				if (kernelAWidth > kernelBWidth)
				{
					minIntegrationLimit = x - m_kernelB.positive_support.x;
					maxIntegrationLimit = x - (-m_kernelB.negative_support.x);
				}
				else
				{
					minIntegrationLimit = -m_kernelA.negative_support.x;
					maxIntegrationLimit = m_kernelA.positive_support.x;
				}
			}
			else if ((x > (kernelWideMaxSupport + kernelNarrowMinSupport)) && (x <= (kernelWideMaxSupport + kernelNarrowMaxSupport)))
			{
				minIntegrationLimit = x - m_kernelB.positive_support.x;
				maxIntegrationLimit = m_kernelA.positive_support.x;
			}
		}
		assert(minIntegrationLimit <= maxIntegrationLimit);

		return { minIntegrationLimit, maxIntegrationLimit};
	}

	static inline float getKernelWidth(const IImageFilterKernel& kernel)
	{
		return kernel.negative_support.x + kernel.positive_support.x;
	};

	const KernelA m_kernelA;
	const KernelB m_kernelB;
};

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CBoxImageFilterKernel>, CScaledImageFilterKernel<CBoxImageFilterKernel>>::weight(const float x, const uint32_t channel, const uint32_t) const;

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CKaiserImageFilterKernel<>>, CScaledImageFilterKernel<CKaiserImageFilterKernel<>>>::weight(const float x, const uint32_t channel, const uint32_t) const;

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CGaussianImageFilterKernel<>>, CScaledImageFilterKernel<CGaussianImageFilterKernel<>>>::weight(const float x, const uint32_t channel, const uint32_t) const;

// TODO: Specializations of CConvolutionImageFilterKernel for <CScaledImageFilterKernel<Triangle>,CScaledImageFilterKernel<Triangle>> = this is tricky but feasible

// Dirac with Box

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CDiracImageFilterKernel>, CScaledImageFilterKernel<CBoxImageFilterKernel>>::weight(const float x, const uint32_t channel, const uint32_t) const;

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CBoxImageFilterKernel>, CScaledImageFilterKernel<CDiracImageFilterKernel>>::weight(const float x, const uint32_t channel, const uint32_t) const;

// Dirac with Triangle

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CDiracImageFilterKernel>, CScaledImageFilterKernel<CTriangleImageFilterKernel>>::weight(const float x, const uint32_t channel, const uint32_t) const;

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CTriangleImageFilterKernel>, CScaledImageFilterKernel<CDiracImageFilterKernel>>::weight(const float x, const uint32_t channel, const uint32_t) const;

// Dirac with Gaussian

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CDiracImageFilterKernel>, CScaledImageFilterKernel<CGaussianImageFilterKernel<>>>::weight(const float x, const uint32_t channel, const uint32_t) const;

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CGaussianImageFilterKernel<>>, CScaledImageFilterKernel<CDiracImageFilterKernel>>::weight(const float x, const uint32_t channel, const uint32_t) const;

// Dirac with Mitchell

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CDiracImageFilterKernel>, CScaledImageFilterKernel<CMitchellImageFilterKernel<>>>::weight(const float x, const uint32_t channel, const uint32_t) const;

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CMitchellImageFilterKernel<>>, CScaledImageFilterKernel<CDiracImageFilterKernel>>::weight(const float x, const uint32_t channel, const uint32_t) const;

// Dirac with Kaiser

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CDiracImageFilterKernel>, CScaledImageFilterKernel<CKaiserImageFilterKernel<>>>::weight(const float x, const uint32_t channel, const uint32_t) const;

template <>
float CConvolutionImageFilterKernel<CScaledImageFilterKernel<CKaiserImageFilterKernel<>>, CScaledImageFilterKernel<CDiracImageFilterKernel>>::weight(const float x, const uint32_t channel, const uint32_t) const;

} // end namespace nbl::asset

#endif