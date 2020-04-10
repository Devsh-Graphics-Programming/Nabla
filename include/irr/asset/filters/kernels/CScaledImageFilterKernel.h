// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_SCALED_IMAGE_FILTER_KERNEL_H_INCLUDED__
#define __IRR_C_SCALED_IMAGE_FILTER_KERNEL_H_INCLUDED__

#include "irr/core/core.h"

#include "irr/asset/filters/kernels/IImageFilterKernel.h"

#include <type_traits>

namespace irr
{
namespace asset
{

namespace impl
{

class CScaledImageFilterKernelBase
{
	public:
		CScaledImageFilterKernelBase(const core::vectorSIMDf& _rscale) : rscale(_rscale.x,_rscale.y,_rscale.z,_rscale.x*_rscale.y*_rscale.z) {}

		const core::vectorSIMDf rscale;
};

}

template<class Kernel>
class CScaledImageFilterKernel : private Kernel, public impl::CScaledImageFilterKernelBase, public CImageFilterKernel<CScaledImageFilterKernel<Kernel>,typename Kernel::value_type>
{
	public:
		using value_type = typename Kernel::value_type;

		_IRR_STATIC_INLINE_CONSTEXPR bool is_separable = Kernel::is_separable;

		CScaledImageFilterKernel(const float* _scale, Kernel&& k=Kernel()) : Kernel(std::move(k)),
			impl::CScaledImageFilterKernelBase(core::vectorSIMDf(1.f)/core::vectorSIMDf(_scale[0],_scale[1],_scale[2],1.f)),
			CImageFilterKernel<CScaledImageFilterKernel<Kernel>,value_type>(
					{Kernel::positive_support[0]*_scale[0],Kernel::positive_support[1]*_scale[1],Kernel::positive_support[2]*_scale[2]},
					{Kernel::negative_support[0]*_scale[0],Kernel::negative_support[1]*_scale[1],Kernel::negative_support[2]*_scale[2]}
				)
		{
		}

		static inline bool validate(ICPUImage* inImage, ICPUImage* outImage)
		{
			return Kernel::validate(inImage,outImage);
		}

		template<class PerSampleFunctor=default_sample_functor_t>
		inline void evaluate(value_type* windowData, const core::vectorSIMDf& inPos, const PerSampleFunctor& perSample=PerSampleFunctor()) const
		{
			// TODO: refactor default `evaluate` so it iterates over the window
			// TODO: refactor the per sample functors so they get applied with other functors embedded and also take a window position
			// TODO: implement scaling here, as:
			// 1. Modifying the window
			// 2. Wrapping the per-sample functor and intervening with a scale to the input
			Kernel::evaluate(windowData,inPos*rscale,perSample);
			for (auto i=0; i<4; i++)
				out[i] *= rscale.w;
		}
};


} // end namespace asset
} // end namespace irr

#endif