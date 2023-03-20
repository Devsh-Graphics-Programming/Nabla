// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_CONVOLUTION_IMAGE_FILTER_KERNEL_H_INCLUDED__
#define __NBL_ASSET_C_CONVOLUTION_IMAGE_FILTER_KERNEL_H_INCLUDED__

#include "nbl/asset/filters/kernels/CDiracImageFilterKernel.h"
#include "nbl/asset/filters/kernels/CBoxImageFilterKernel.h"
#include "nbl/asset/filters/kernels/CTriangleImageFilterKernel.h"
#include "nbl/asset/filters/kernels/CGaussianImageFilterKernel.h"
#include "nbl/asset/filters/kernels/CKaiserImageFilterKernel.h"
#include "nbl/asset/filters/kernels/CMitchellImageFilterKernel.h"

namespace nbl::asset
{

// this is the horribly slow generic version that you should not use (only use the specializations)
template<typename KernelA, typename KernelB>
class CConvolutionImageFilterKernel : public CFloatingPointSeparableImageFilterKernelBase<CConvolutionImageFilterKernel<KernelA, KernelB>>
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

		float kernelWideWidth = getKernelWidth(m_kernelA);
		float kernelNarrowWidth = getKernelWidth(m_kernelB);

		if (kernelWideWidth < kernelNarrowWidth)
		{
			std::swap(kernelNarrowMinSupport, kernelWideMinSupport);
			std::swap(kernelNarrowMaxSupport, kernelWideMaxSupport);
			std::swap(kernelNarrowWidth, kernelWideWidth);
		}

		const float kernelNarrowWidth_half = kernelNarrowWidth * 0.5;

		double minIntegrationLimit = 0.0, maxIntegrationLimit = 0.0;
		{
			if ((x >= (kernelWideMinSupport - kernelNarrowWidth_half)) && (x <= (kernelWideMinSupport + kernelNarrowWidth_half)))
			{
				minIntegrationLimit = kernelWideMinSupport;
				maxIntegrationLimit = x + kernelNarrowWidth_half;
			}
			else if ((x >= (kernelWideMinSupport + kernelNarrowWidth_half)) && (x <= (kernelWideMaxSupport - kernelNarrowWidth_half)))
			{
				minIntegrationLimit = x - kernelNarrowWidth_half;
				maxIntegrationLimit = x + kernelNarrowWidth_half;
			}
			else if ((x >= (kernelWideMaxSupport - kernelNarrowWidth_half)) && (x <= (kernelWideMaxSupport + kernelNarrowWidth_half)))
			{
				minIntegrationLimit = x - kernelNarrowWidth_half;
				maxIntegrationLimit = kernelWideMaxSupport;
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
float CConvolutionImageFilterKernel<CBoxImageFilterKernel, CBoxImageFilterKernel>::weight(const float x, const uint32_t channel, const uint32_t) const;

template <>
float CConvolutionImageFilterKernel<CGaussianImageFilterKernel, CGaussianImageFilterKernel>::weight(const float x, const uint32_t channel, const uint32_t) const;

// TODO: Specialization: CConvolutionImageFilterKernel<Triangle,Triangle> = this is tricky but feasible

template <>
float CConvolutionImageFilterKernel<CKaiserImageFilterKernel, CKaiserImageFilterKernel>::weight(const float x, const uint32_t channel, const uint32_t) const;

// Dirac with Box

template <>
float CConvolutionImageFilterKernel<CDiracImageFilterKernel, CBoxImageFilterKernel>::weight(const float x, const uint32_t channel, const uint32_t) const;

template <>
float CConvolutionImageFilterKernel<CBoxImageFilterKernel, CDiracImageFilterKernel>::weight(const float x, const uint32_t channel, const uint32_t) const;

// Dirac with Triangle

template <>
float CConvolutionImageFilterKernel<CDiracImageFilterKernel, CTriangleImageFilterKernel>::weight(const float x, const uint32_t channel, const uint32_t) const;

template <>
float CConvolutionImageFilterKernel<CTriangleImageFilterKernel, CDiracImageFilterKernel>::weight(const float x, const uint32_t channel, const uint32_t) const;

// Dirac with Gaussian

template <>
float CConvolutionImageFilterKernel<CDiracImageFilterKernel, CGaussianImageFilterKernel>::weight(const float x, const uint32_t channel, const uint32_t) const;

template <>
float CConvolutionImageFilterKernel<CGaussianImageFilterKernel, CDiracImageFilterKernel>::weight(const float x, const uint32_t channel, const uint32_t) const;

// Dirac with Mitchell

template <>
float CConvolutionImageFilterKernel<CDiracImageFilterKernel, CMitchellImageFilterKernel<>>::weight(const float x, const uint32_t channel, const uint32_t) const;

template <>
float CConvolutionImageFilterKernel<CMitchellImageFilterKernel<>, CDiracImageFilterKernel>::weight(const float x, const uint32_t channel, const uint32_t) const;

// Dirac with Kaiser

template <>
float CConvolutionImageFilterKernel<CDiracImageFilterKernel, CKaiserImageFilterKernel>::weight(const float x, const uint32_t channel, const uint32_t) const;

template <>
float CConvolutionImageFilterKernel<CKaiserImageFilterKernel, CDiracImageFilterKernel>::weight(const float x, const uint32_t channel, const uint32_t) const;

} // end namespace nbl::asset

#endif