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
		using StaticPolymorphicBase = CImageFilterKernel<CScaledImageFilterKernel<Kernel>, typename Kernel::value_type>;

	public:
		_IRR_STATIC_INLINE_CONSTEXPR auto MaxChannels = Kernel::MaxChannels;
		using value_type = typename Kernel::value_type;

		_IRR_STATIC_INLINE_CONSTEXPR bool is_separable = Kernel::is_separable;

		CScaledImageFilterKernel(const core::vectorSIMDf& _scale, Kernel&& k=Kernel()) : Kernel(std::move(k)),
			impl::CScaledImageFilterKernelBase(core::vectorSIMDf(1.f,1.f,1.f,0.f).preciseDivision(_scale)),
			StaticPolymorphicBase(
					{Kernel::positive_support[0]*_scale[0],Kernel::positive_support[1]*_scale[1],Kernel::positive_support[2]*_scale[2]},
					{Kernel::negative_support[0]*_scale[0],Kernel::negative_support[1]*_scale[1],Kernel::negative_support[2]*_scale[2]}
				)
		{
		}
		CScaledImageFilterKernel(const core::vectorSIMDf& _scale, const Kernel& k=Kernel()) : CScaledImageFilterKernel(_scale,Kernel(k)) {}

		template<typename... Args>
		inline core::vectorSIMDi32 getWindowMinCoord(Args&&... args) const
		{
			return StaticPolymorphicBase::getWindowMinCoord(std::forward<Args>(args)...);
		}

		inline int32_t getWindowVolume() const
		{
			return StaticPolymorphicBase::getWindowVolume();
		}


		static inline bool validate(ICPUImage* inImage, ICPUImage* outImage)
		{
			return Kernel::validate(inImage,outImage);
		}
		
		template<class PreFilter=const typename StaticPolymorphicBase::default_sample_functor_t, class PostFilter=const typename StaticPolymorphicBase::default_sample_functor_t>
		inline void evaluate(const core::vectorSIMDf& globalPos, PreFilter& preFilter, PostFilter& postFilter) const
		{
			StaticPolymorphicBase::evaluate<PreFilter,PostFilter>(globalPos,preFilter,postFilter);
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