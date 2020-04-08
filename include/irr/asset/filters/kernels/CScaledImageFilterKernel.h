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

		CScaledImageFilterKernel(float _scale[3]) : Kernel(k),
			impl::CScaledImageFilterKernelBase(core::vectorSIMDf(1.f)/core::vectorSIMDf(_scale[0],_scale[1],_scale[2],1.f)),
			CImageFilterKernel<CScaledImageFilterKernel<Kernel>,value_type>(
					{Kernel::positive_support[0]*rscale[0],Kernel::positive_support[1]*rscale[1],Kernel::positive_support[2]*rscale[2]},
					{Kernel::negative_support[0]*rscale[0],Kernel::negative_support[1]*rscale[1],Kernel::negative_support[2]*rscale[2]}
				)
		{
		}

		static inline bool validate(ICPUImage* inImage, ICPUImage* outImage)
		{
			return Kernel::validate(inImage,outImage);
		}
/*
		inline float weight(const core::vectorSIMDf& inPos)
		{
			return rscale*Kernel::weight(inPos*rscale);
		}
*/
		inline void evaluate(value_type* out,const core::vectorSIMDf& inPos, const value_type*** slices) const
		{
			Kernel::evaluate(out,inPos*rscale,slices);
			for (auto i=0; i<4; i++)
				out[i] *= rscale.w;
		}
};


} // end namespace asset
} // end namespace irr

#endif