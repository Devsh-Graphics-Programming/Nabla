// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_SCALED_IMAGE_FILTER_KERNEL_H_INCLUDED__
#define __NBL_ASSET_C_SCALED_IMAGE_FILTER_KERNEL_H_INCLUDED__

#include "nbl/core/declarations.h"

#include "nbl/asset/filters/kernels/IImageFilterKernel.h"

#include <type_traits>

namespace nbl
{
namespace asset
{

namespace impl
{
	
template<class Kernel>
class NBL_API CScaledImageFilterKernelBase
{
	public:
		// we preserve all basic properties of the original kernel
		_NBL_STATIC_INLINE_CONSTEXPR auto MaxChannels = Kernel::MaxChannels;
		using value_type = typename Kernel::value_type;

		_NBL_STATIC_INLINE_CONSTEXPR bool is_separable = Kernel::is_separable;

		// constructor
		CScaledImageFilterKernelBase(const core::vectorSIMDf& _rscale, Kernel&& k) : kernel(std::move(k)), rscale(_rscale.x,_rscale.y,_rscale.z,1.f), userData(_rscale.x*_rscale.y*_rscale.z) {}

		Kernel kernel;
		// reciprocal of the scale, the w component holds the scale that needs to be applied to the kernel values to preserve the integral
		// 1/(A*B*C) InfiniteIntegral f(x/A,y/B,z/C) dx dy dz == InfiniteIntegral f(x,y,z) dx dy dz
		const core::vectorSIMDf rscale;

protected:
	const IImageFilterKernel::ScaleFactorUserData userData;
};

}

// this kernel will become a stretched version of the original kernel while keeping its integral constant
template<class Kernel>
class NBL_API CScaledImageFilterKernel : //order of bases is important! do not change
	public impl::CScaledImageFilterKernelBase<Kernel>, public CImageFilterKernel<CScaledImageFilterKernel<Kernel>,typename impl::CScaledImageFilterKernelBase<Kernel>::value_type>
{
		using Base = impl::CScaledImageFilterKernelBase<Kernel>;
		using StaticPolymorphicBase = CImageFilterKernel<CScaledImageFilterKernel<Kernel>,typename Base::value_type>;

	public:
		_NBL_STATIC_INLINE_CONSTEXPR auto MaxChannels = Base::MaxChannels;
		using value_type = typename Base::value_type;

		_NBL_STATIC_INLINE_CONSTEXPR bool is_separable = Base::is_separable;

		// the scale is how much we want to stretch the support, so if we have a box function kernel with support -0.5,0.5 then scaling it with `_scale=4.0`
		// would give us a kernel with support -2.0,2.0 which still has the same area under the curve (integral)
		CScaledImageFilterKernel(const core::vectorSIMDf& _scale, Kernel&& k=Kernel()) : Base(core::vectorSIMDf(1.f).preciseDivision(_scale),std::move(k)),
			StaticPolymorphicBase(
					{Base::kernel.positive_support[0]*_scale[0],Base::kernel.positive_support[1]*_scale[1],Base::kernel.positive_support[2]*_scale[2]},
					{Base::kernel.negative_support[0]*_scale[0],Base::kernel.negative_support[1]*_scale[1],Base::kernel.negative_support[2]*_scale[2]}
				)
		{
		}
		// overload for uniform scale in all dimensions
		CScaledImageFilterKernel(const core::vectorSIMDf& _scale, const Kernel& k=Kernel()) : CScaledImageFilterKernel(_scale,Kernel(k)) {}

		// make sure we let everyone know we changed the domain of the function by stretching it
		inline const IImageFilterKernel::UserData* getUserData() const { return &(Base::userData); }

		// the validation usually is not support dependent, its usually about the input/output formats of an image, etc. so we use old Kernel validation
		static inline bool validate(ICPUImage* inImage, ICPUImage* outImage)
		{
			//is validate() always static?
			return Kernel::validate(inImage, outImage);
		}

		// TODO(achal): Make private.
		inline float weight(const float x, const uint32_t channel) const
		{
			// This will breakdown if `negative_support` didn't start a negative value.
			const bool inDomain = ((-x) <= this->negative_support.x) && (x <= this->positive_support.x);
			return inDomain ? this->kernel.weight(x*this->rscale.x, channel) * this->rscale.x : 0.f;
		}

		// this is the only bit that differs
		template<class PreFilter, class PostFilter>
		struct sample_functor_t
		{
				sample_functor_t(const CScaledImageFilterKernel<Kernel>* _this, PreFilter& _preFilter, PostFilter& _postFilter) :
					_this(_this), preFilter(_preFilter), postFilter(_postFilter) {}

				// so this functor wraps the original one of the unscaled in a peculiar way
				inline void operator()(value_type* windowSample, core::vectorSIMDf& relativePos, const core::vectorSIMDi32& globalTexelCoord, const IImageFilterKernel::UserData* userData)
				{
					// it actually injects an extra preFilter functor after the original to rescale the `relativePos`
					auto wrap = [this](value_type* windowSample, core::vectorSIMDf& relativePos, const core::vectorSIMDi32& globalTexelCoord, const IImageFilterKernel::UserData* userData)
					{
						preFilter(windowSample,relativePos,globalTexelCoord,userData);
						relativePos *= _this->rscale;
					};
					// inject the wrap as a pre-filter instead of the original in the non-scaled kernel functor
					_this->kernel.create_sample_functor_t(wrap,postFilter)(windowSample,relativePos,globalTexelCoord,userData);
				}

			private:
				const CScaledImageFilterKernel<Kernel>* _this;
				PreFilter& preFilter;
				PostFilter& postFilter;
		};

		// the method all kernels must define and overload
		template<class PreFilter, class PostFilter>
		inline auto create_sample_functor_t(PreFilter& preFilter, PostFilter& postFilter) const
		{
			return sample_functor_t<PreFilter,PostFilter>(this,preFilter,postFilter);
		}

		// need this to resolve to correct base
		NBL_DECLARE_DEFINE_CIMAGEFILTER_KERNEL_PASS_THROUGHS(StaticPolymorphicBase)
};


} // end namespace asset
} // end namespace nbl

#endif