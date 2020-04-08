// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_KERNELS_H_INCLUDED__
#define __IRR_KERNELS_H_INCLUDED__


#include "irr/asset/filters/kernels/IImageFilterKernel.h"
#include "irr/asset/filters/kernels/CommonImageFilterKernels.h"
/*
#include "irr/asset/filters/kernels/CScaledImageFilterKernel.h"

namespace irr
{
namespace asset
{
	

// caches weights
template<class Kernel>
class CMultiphaseKernel : public CImageFilterKernel<CMultiphaseKernel<Kernel> >, private Kernel
{
	public:
		_IRR_STATIC_INLINE_CONSTEXPR bool is_separable = Kernel;

		CMultiphaseKernel(Kernel&& k) : Kernel(std::move(k)
		{
		}
		
	protected:
		static inline core::vectorSIMDu32 computePhases(const core::vectorSIMDu32& from, const core::vectorSIMDu32& to)
		{
			assert(!(to>from).any()); // Convolution Kernel cannot be used for upscaling!
			return from/core::gcd(to,from);
		}
		static inline uint32_t computePhaseStorage(const core::vectorSIMDu32& from, const core::vectorSIMDu32& to)
		{
			auto phases = computePhases(from,to);
			auto samplesInSupports = ceil();
			if constexpr(is_separable)
			{

			}
		}
};

template<class KernelA, class KernelB>
class CKernelConvolution : public CImageFilterKernel<CKernelConvolution<KernelA,KernelB> >, private KernelA, private KernelB
{
	public:
		_IRR_STATIC_INLINE_CONSTEXPR bool is_separable = KernelA::is_separable&&KernelB::is_separable;
		static_assert(is_separable,"Convolving Non-Separable Filters is a TODO!");

		const float positive_support[3];
		const float negative_support[3];
		CKernelConvolution(KernelA&& a, KernelB&& b) : KernelA(std::move(a)), KernelB(std::move(b)),
			positive_support({
								KernelA::positive_support[0]+KernelB::positive_support[0],
								KernelA::positive_support[1]+KernelB::positive_support[1],
								KernelA::positive_support[2]+KernelB::positive_support[2]
							}),
			negative_support({
								KernelA::negative_support[0]+KernelB::negative_support[0],
								KernelA::negative_support[1]+KernelB::negative_support[1],
								KernelA::negative_support[2]+KernelB::negative_support[2]
							})
		{}

	
		static inline bool validate(ICPUImage* inImage, ICPUImage* outImage)
		{
			return KernelA::validate(inImage,outImage) && KernelB::validate(inImage,outImage);
		}

		inline float evaluate(const core::vectorSIMDf& inPos, uint32_t iterations=64u)
		{
			const double dx = (positive_support[0]-negative_support[0])/double(iterations);
			double sum = 0.0;
			for (uint32_t i=0u; i<iterations; i++)
			{
				auto fakePos = ;
				sum += KernelA::evaluate(fakePos)*KernelB::evaluate(inPos-fakePos);
			}
			return sum;
		}
};

} // end namespace asset
} // end namespace irr

*/

#endif