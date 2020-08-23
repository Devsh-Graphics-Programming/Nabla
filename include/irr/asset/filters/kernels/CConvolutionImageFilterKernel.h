// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_CONVOLUTION_IMAGE_FILTER_KERNEL_H_INCLUDED__
#define __IRR_C_CONVOLUTION_IMAGE_FILTER_KERNEL_H_INCLUDED__

#include "irr/asset/filters/kernels/CBoxImageFilterKernel.h"
#include "irr/asset/filters/kernels/CTriangleImageFilterKernel.h"
#include "irr/asset/filters/kernels/CGaussianImageFilterKernel.h"
#include "irr/asset/filters/kernels/CKaiserImageFilterKernel.h"
#include "irr/asset/filters/kernels/CMitchellImageFilterKernel.h"

namespace irr
{
namespace asset
{

#if 0 //TODO: Implement later
// class for an image filter kernel which is a convolution of two image filter kernels
template<class KernelA, class KernelB>
class CConvolutionImageFilterKernel;

namespace impl
{
	template<class KernelA, class KernelB, class CRTP>
	class CConvolutionImageFilterKernelBase : protected KernelA, protected KernelB, public CImageFilterKernel<CConvolutionImageFilterKernelBase<KernelA,KernelB,CRTP>, typename KernelA::value_type>
	{
			static_assert(std::is_same_v<typename KernelA::value_type,typename KernelB::value_type>::value, "Both kernels must use the same value_type!");
			struct priv_t
			{
				const float positive_support[3];
				const float negative_support[3];
			};
			const priv;

		protected:
			CConvolutionImageFilterKernelBase(KernelA&& a, KernelB&& b) : KernelA(std::move(a)), KernelB(std::move(b)),
				IImageFilterKernel{
						{
							KernelA::positive_support[0]+KernelB::positive_support[0],
							KernelA::positive_support[1]+KernelB::positive_support[1],
							KernelA::positive_support[2]+KernelB::positive_support[2]
						},
						{
							KernelA::negative_support[0]+KernelB::negative_support[0],
							KernelA::negative_support[1]+KernelB::negative_support[1],
							KernelA::negative_support[2]+KernelB::negative_support[2]
						}
					}
			{
			}

		public:
			_IRR_STATIC_INLINE_CONSTEXPR bool is_separable = KernelA::is_separable&&KernelB::is_separable;

			static inline bool validate(ICPUImage* inImage, ICPUImage* outImage)
			{
				return KernelA::validate(inImage, outImage) && KernelB::validate(inImage, outImage);
			}

			inline float evaluate(const core::vectorSIMDf& inPos)
			{
				return CRTP::evaluate
			}
	};
}

/*

TODO: Specializations of CConvolutionImageFilterKernel
<A,B> -> <CScaledImageFilterKernel<A>,CScaledImageFilterKernel<B>>  but only if both A and B are derived from `CFloatingPointIsotropicSeparableImageFilterKernelBase`

<CScaledImageFilterKernel<Kaiser>,CScaledImageFilterKernel<Kaiser>> = just pick the wider kaiser
<CScaledImageFilterKernel<Gaussian>,CScaledImageFilterKernel<Gaussian>> = just add the stardard deviations together
<CScaledImageFilterKernel<Box>,CScaledImageFilterKernel<Box>> = you need to find the area between both boxes

<CScaledImageFilterKernel<Triangle>,CScaledImageFilterKernel<Triangle>> = this is tricky but feasible

// these I eventually want for perfect mip-maps (probably only as tabulated polyphase stuff)
<CScaledImageFilterKernel<Kaiser>,CScaledImageFilterKernel<Mitchell>>
<CScaledImageFilterKernel<Gaussian>,CScaledImageFilterKernel<Mitchell>>
<CScaledImageFilterKernel<Kaiser>,CScaledImageFilterKernel<Gaussian>>
*/

// this is the horribly slow generic version that you should not use (only use the specializations)
template<class KernelA, class KernelB>
class CConvolutionImageFilterKernel : public impl::CConvolutionImageFilterKernelBase<KernelA,KernelB,CConvolutionImageFilterKernel<KernelA,KernelB> >
{
		using Base = CConvolutionImageFilterKernelBase<KernelA,KernelB,CConvolutionImageFilterKernel<KernelA,KernelB> >;

	public:
		using Base::Base;

		// this is a special implementation for general 
		static_assert(CConvolutionImageFilterKernelBase<KernelA, KernelB>::is_separable, "Convolving Non-Separable Filters is a TODO!");
		inline float evaluate(const core::vectorSIMDf& inPos, uint32_t iterations=64u)
		{
			#ifdef _IRR_DEBUG
				static_assert(false,"Not Implemented Yet!");
				const double dx = (positive_support[0]-negative_support[0])/double(iterations+1u);
				double sum = 0.0;
				for (uint32_t i=0u; i<iterations; i++)
				{
					auto fakePos = inPos;
					sum += KernelA::evaluate(fakePos)*KernelB::evaluate(inPos-fakePos);
				}
				return sum/double(iterations);
			#else
				static_assert(false,"You shouldn't be using this in production!");
			#endif
		}
};
#endif

} // end namespace asset
} // end namespace irr

#endif