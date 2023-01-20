// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_COMMON_IMAGE_FILTER_KERNELS_H_INCLUDED__
#define __NBL_ASSET_COMMON_IMAGE_FILTER_KERNELS_H_INCLUDED__


#include "nbl/asset/filters/kernels/IImageFilterKernel.h"

#include <ratio>

namespace nbl
{
namespace asset
{

// base class for all kernels that require the pixels and arithmetic to be done in precise floats
class NBL_API CFloatingPointOnlyImageFilterKernelBase
{
	public:
		using value_type = double; // should probably allot `float`s at some point

		static inline bool validate(ICPUImage* inImage, ICPUImage* outImage)
		{
			const auto& inParams = inImage->getCreationParameters();
			const auto& outParams = inImage->getCreationParameters();
			return !(isIntegerFormat(inParams.format)||isIntegerFormat(outParams.format));
		}

	protected:
		CFloatingPointOnlyImageFilterKernelBase() {}
};

// base class for all kernels which can be separated into axis-aligned passes
class NBL_API CSeparableImageFilterKernelBase
{
	public:
		_NBL_STATIC_INLINE_CONSTEXPR bool is_separable = true;

	protected:
		CSeparableImageFilterKernelBase() {}
};

// base class for all kernels that require pixels and arithmetic to be done in precise floats AND are separable AND have the same kernel function in each dimension AND have a rational support
template<class CRTP>
class NBL_API CFloatingPointSeparableImageFilterKernelBase : public CImageFilterKernel<CRTP,typename CFloatingPointOnlyImageFilterKernelBase::value_type>, public CFloatingPointOnlyImageFilterKernelBase, public CSeparableImageFilterKernelBase
{
	public:
		using value_type = typename CFloatingPointOnlyImageFilterKernelBase::value_type;
		_NBL_STATIC_INLINE_CONSTEXPR auto MaxChannels = 4;

	private:
		using StaticPolymorphicBase = CImageFilterKernel<CRTP,value_type>;

	public:
		CFloatingPointSeparableImageFilterKernelBase(float _negative_support, float _positive_support) : StaticPolymorphicBase({_negative_support,_negative_support,_negative_support},{_positive_support,_positive_support,_positive_support}) {}

		//
		template<class PreFilter, class PostFilter>
		struct sample_functor_t
		{
				sample_functor_t(const CRTP* __this, PreFilter& _preFilter, PostFilter& _postFilter) :
					_this(__this), preFilter(_preFilter), postFilter(_postFilter) {}

				inline void operator()(value_type* windowSample, core::vectorSIMDf& relativePos, const core::vectorSIMDi32& globalTexelCoord, const core::vectorSIMDf& scale);

			private:
				const CRTP* _this;
				PreFilter& preFilter;
				PostFilter& postFilter;
		};

		// this is the function that must be defined for each kernel
		template<class PreFilter, class PostFilter>
		inline auto create_sample_functor_t(PreFilter& preFilter, PostFilter& postFilter) const
		{
			return sample_functor_t<PreFilter,PostFilter>(static_cast<const CRTP*>(this),preFilter,postFilter);
		}

		// derived classes must declare this
		inline float weight(float x, int32_t channel) const
		{
			return static_cast<const CRTP*>(this)->weight(x,channel);
		}

	protected:
		// utility function so we dont evaluate `weight` function in children outside the support and just are able to return 0.f
		inline bool inDomain(float x) const
		{
			return (-x)<=StaticPolymorphicBase::negative_support.x && x<StaticPolymorphicBase::positive_support.x;
		}
};


// base class for all kernels that have the same support in each dimension AND have a rational support
// there's nothing special about having a rational support, we just use that type because there's no possibility of using `float` as a template parameter in C++
template<class Support=std::ratio<1,1> >
class NBL_API CIsotropicImageFilterKernelBase
{
	public:
		using isotropic_support_as_ratio = Support;
	protected:
		_NBL_STATIC_INLINE_CONSTEXPR float isotropic_support = float(Support::num)/float(Support::den);
};

// same as CFloatingPointSeparableImageFilterKernelBase but with added constraint that support is symmetric around the orign
template<class CRTP,class Support=std::ratio<1,1> >
class NBL_API CFloatingPointIsotropicSeparableImageFilterKernelBase :	public CFloatingPointSeparableImageFilterKernelBase<CFloatingPointIsotropicSeparableImageFilterKernelBase<CRTP,Support>>,
																public CIsotropicImageFilterKernelBase<Support>
{
		using Base =  CFloatingPointSeparableImageFilterKernelBase<CFloatingPointIsotropicSeparableImageFilterKernelBase<CRTP,Support>>;
		using Base2 = CIsotropicImageFilterKernelBase<Support>;
	protected:
		_NBL_STATIC_INLINE_CONSTEXPR float isotropic_support = Base2::isotropic_support;

	public:
		CFloatingPointIsotropicSeparableImageFilterKernelBase() : Base(isotropic_support,isotropic_support) {}

		// need this declared to forward
		inline float weight(float x, int32_t channel) const
		{
			return static_cast<const CRTP*>(this)->weight(x, channel);
		}
};

} // end namespace asset
} // end namespace nbl

#endif