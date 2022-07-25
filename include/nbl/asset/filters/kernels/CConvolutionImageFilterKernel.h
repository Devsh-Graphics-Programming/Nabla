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

namespace nbl
{
namespace asset
{

//during implementation remember about composition instead of inheritance
#if 0 // implementations are a TODO (we probably need a polyphase kernel to cache these results)
// class for an image filter kernel which is a convolution of two image filter kernels
template<class KernelA, class KernelB>
class NBL_API CConvolutionImageFilterKernel;

namespace impl
{
	template<class KernelA, class KernelB, class CRTP>
	class NBL_API CConvolutionImageFilterKernelBase : protected KernelA, protected KernelB, public CFloatingPointSeparableImageFilterKernelBase<CConvolutionImageFilterKernelBase<KernelA,KernelB,CRTP>>
	{
		// TODO: figure out what I meant to do here
			//static_assert(std::is_same_v<typename KernelA::value_type,typename KernelB::value_type>::value, "Both kernels must use the same value_type!");
			static_assert(
				std::is_base_of<CFloatingPointIsotropicSeparableImageFilterKernelBase<KernelA>,typename KernelA>::value&&
				std::is_base_of<CFloatingPointIsotropicSeparableImageFilterKernelBase<KernelB>,typename KernelB>::value,
				"Both kernels must be derived from CFloatingPointIsotropicSeparableImageFilterKernelBase!"
			);

		protected:
			CConvolutionImageFilterKernelBase(KernelA&& a, KernelB&& b) : KernelA(std::move(a)), KernelB(std::move(b)),
				CFloatingPointSeparableImageFilterKernelBase<CConvolutionImageFilterKernelBase<KernelA, KernelB, CRTP>>{
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
			_NBL_STATIC_INLINE_CONSTEXPR bool is_separable = KernelA::is_separable&&KernelB::is_separable;

			static inline bool validate(ICPUImage* inImage, ICPUImage* outImage)
			{
				return KernelA::validate(inImage, outImage) && KernelB::validate(inImage, outImage);
			}

			static_assert(CConvolutionImageFilterKernelBase<KernelA, KernelB>::is_separable, "Convolving Non-Separable Filters is a TODO!");
			// specialization defines this
			inline float weight(float x, int32_t channel) const
			{
				return CRTP::weight(x,channel);
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
class NBL_API CConvolutionImageFilterKernel : public impl::CConvolutionImageFilterKernelBase<KernelA,KernelB,CConvolutionImageFilterKernel<KernelA,KernelB> >
{
		using Base = CConvolutionImageFilterKernelBase<KernelA,KernelB,CConvolutionImageFilterKernel<KernelA,KernelB> >;

	public:
		using Base::Base;

		// this is a special implementation for general 
		inline float weight(const float x, int32_t channel, uint32_t iterations=64u)
		{
			static_assert(false,"Not Implemented Yet!");
			const double dx = (positive_support[0]-negative_support[0])/double(iterations+1u);
			double sum = 0.0;
			for (uint32_t i=0u; i<iterations; i++)
			{
				auto fakePos = inPos;
				sum += KernelA::evaluate(fakePos)*KernelB::evaluate(inPos-fakePos);
			}
			return sum/double(iterations);
		}
};
#endif

} // end namespace asset
} // end namespace nbl

#endif