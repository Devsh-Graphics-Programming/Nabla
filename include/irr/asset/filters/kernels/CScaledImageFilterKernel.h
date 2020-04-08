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


template<class Kernel>
class CScaledImageFilterKernel : public CImageFilterKernel<CScaledImageFilterKernel<Kernel> >, private Kernel
{
	public:
		_IRR_STATIC_INLINE_CONSTEXPR bool is_separable = Kernel::is_separable;
		const core::vectorSIMDf rscale;
		const float positive_support[3];
		const float negative_support[3];
		CScaledImageFilterKernel(float _scale[3]) : Kernel(k),
			rscale(core::vectorSIMDf(1.f)/core::vectorSIMDf(_scale[0],_scale[1],_scale[2],1.f)),
			positive_support(	{
									Kernel::positive_support[0]*rscale[0],
									Kernel::positive_support[1]*rscale[1],
									Kernel::positive_support[2]*rscale[2]
								}),
			negative_support(	{
									Kernel::negative_support[0]*rscale[0],
									Kernel::negative_support[1]*rscale[1],
									Kernel::negative_support[2]*rscale[2]
								})
		{
		}

		static inline bool validate(ICPUImage* inImage, ICPUImage* outImage)
		{
			return Kernel::validate(inImage,outImage);
		}

		inline float evaluate(const core::vectorSIMDf& inPos)
		{
			return rscale*Kernel::evaluate(inPos*rscale);
		}
};


} // end namespace asset
} // end namespace irr

#endif