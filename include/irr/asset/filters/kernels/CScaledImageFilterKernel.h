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

		// reciprocal of the scale, the w component holds the scale that needs to be applied to the kernel values to preserve the integral
		// 1/(A*B*C) InfiniteIntegral f(x/A,y/B,z/C) dx dy dz == InfiniteIntegral f(x,y,z) dx dy dz
		const core::vectorSIMDf rscale;
};

}

// this kernel will become a stretched version of the original kernel while keeping its integral constant
template<class Kernel>
class CScaledImageFilterKernel : private Kernel, public impl::CScaledImageFilterKernelBase, public CImageFilterKernel<CScaledImageFilterKernel<Kernel>,typename Kernel::value_type>
{
		using StaticPolymorphicBase = CImageFilterKernel<CScaledImageFilterKernel<Kernel>, typename Kernel::value_type>;

	public:
		// we preserve all basic properties of the original kernel
		_IRR_STATIC_INLINE_CONSTEXPR auto MaxChannels = Kernel::MaxChannels;
		using value_type = typename Kernel::value_type;

		_IRR_STATIC_INLINE_CONSTEXPR bool is_separable = Kernel::is_separable;

		// the scale is how much we want to stretch the support, so if we have a box function kernel with support -0.5,0.5 then scaling it with `_scale=4.0`
		// would give us a kernel with support -2.0,2.0 which still has the same area under the curve (integral)
		CScaledImageFilterKernel(const core::vectorSIMDf& _scale, Kernel&& k=Kernel()) : Kernel(std::move(k)),
			impl::CScaledImageFilterKernelBase(core::vectorSIMDf(1.f).preciseDivision(_scale)),
			StaticPolymorphicBase(
					{Kernel::positive_support[0]*_scale[0],Kernel::positive_support[1]*_scale[1],Kernel::positive_support[2]*_scale[2]},
					{Kernel::negative_support[0]*_scale[0],Kernel::negative_support[1]*_scale[1],Kernel::negative_support[2]*_scale[2]}
				)
		{
		}
		// overload for uniform scale in all dimensions
		CScaledImageFilterKernel(const core::vectorSIMDf& _scale, const Kernel& k=Kernel()) : CScaledImageFilterKernel(_scale,Kernel(k)) {}

		// the validation usually is not support dependent, its usually about the input/output formats of an image, etc. so we use old Kernel validation
		static inline bool validate(ICPUImage* inImage, ICPUImage* outImage)
		{
			return Kernel::validate(inImage, outImage);
		}

		// `StaticPolymorphicBase` takes care of all this for us from the newly computed support values
		// we need to use forwarding for some silly compiler scoping reason instead of leaving the function overload undeclared
		template<typename... Args>
		inline core::vectorSIMDi32 getWindowMinCoord(Args&&... args) const
		{
			return StaticPolymorphicBase::getWindowMinCoord(std::forward<Args>(args)...);
		}

		inline const auto& getWindowSize() const
		{
			return StaticPolymorphicBase::getWindowSize();
		}
		
		// we need to use forwarding for some silly compiler scoping reason instead of leaving the function overload undeclared, it basically will do same as the base
		template<class PreFilter=const typename StaticPolymorphicBase::default_sample_functor_t, class PostFilter=const typename StaticPolymorphicBase::default_sample_functor_t>
		inline void evaluate(const core::vectorSIMDf& globalPos, PreFilter& preFilter, PostFilter& postFilter) const
		{
			StaticPolymorphicBase::evaluate(globalPos,preFilter,postFilter);
		}
		template<class PreFilter, class PostFilter>
		inline void evaluateImpl(PreFilter& preFilter, PostFilter& postFilter, value_type* windowSample, core::vectorSIMDf& relativePosAndFactor, const core::vectorSIMDi32& globalTexelCoord) const
		{
			StaticPolymorphicBase::evaluateImpl(preFilter,postFilter,windowSample,relativePosAndFactor,globalTexelCoord);
		}

		template<class PreFilter, class PostFilter>
		struct sample_functor_t
		{
				sample_functor_t(const CScaledImageFilterKernel<Kernel>* __this, PreFilter& _preFilter, PostFilter& _postFilter) :
					_this(__this), preFilter(_preFilter), postFilter(_postFilter) {}

				inline void operator()(value_type* windowSample, core::vectorSIMDf& relativePosAndFactor, const core::vectorSIMDi32& globalTexelCoord)
				{
					auto wrap = [this](value_type* windowSample, core::vectorSIMDf& relativePosAndFactor, const core::vectorSIMDi32& globalTexelCoord)
					{
						preFilter(windowSample,relativePosAndFactor,globalTexelCoord);
						relativePosAndFactor *= _this->rscale;
					};
					static_cast<const Kernel*>(_this)->create_sample_functor_t(wrap,postFilter)(windowSample,relativePosAndFactor,globalTexelCoord);
				}

			private:
				const CScaledImageFilterKernel<Kernel>* _this;
				PreFilter& preFilter;
				PostFilter& postFilter;
		};

		template<class PreFilter, class PostFilter>
		inline auto create_sample_functor_t(PreFilter& preFilter, PostFilter& postFilter) const
		{
			return sample_functor_t(this,preFilter,postFilter);
		}
};


} // end namespace asset
} // end namespace irr

#endif