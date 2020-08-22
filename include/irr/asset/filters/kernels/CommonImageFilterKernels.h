// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_COMMON_IMAGE_FILTER_KERNELS_H_INCLUDED__
#define __IRR_COMMON_IMAGE_FILTER_KERNELS_H_INCLUDED__


#include "irr/asset/filters/kernels/IImageFilterKernel.h"

#include <ratio>

namespace irr
{
namespace asset
{

// base class for all kernels that require the pixels and arithmetic to be done in precise floats
class CFloatingPointOnlyImageFilterKernelBase
{
	public:
		using value_type = double;

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
class CSeparableImageFilterKernelBase
{
	public:
		_IRR_STATIC_INLINE_CONSTEXPR bool is_separable = true;

	protected:
		CSeparableImageFilterKernelBase() {}
};

// base class for all kernels that have the same support in each dimension AND have a rational support
// there's nothing special about having a rational support, we just use that type because there's no possibility of using `float` as a template parameter in C++
template<class Support=std::ratio<1,1> >
class CIsotropicImageFilterKernelBase
{
	protected:
		_IRR_STATIC_INLINE_CONSTEXPR float isotropic_support = float(Support::num)/float(Support::den);
		// utility constexpr array so we can pass a pointer to the base's constructor
		_IRR_STATIC_INLINE_CONSTEXPR float symmetric_support[3] = { isotropic_support,isotropic_support,isotropic_support };
};

// base class for all kernels that require pixels and arithmetic to be done in precise floats AND are separable AND have the same kernel function and support in each dimension AND have a rational support
template<class CRTP,class Support=std::ratio<1,1> >
class CFloatingPointIsotropicSeparableImageFilterKernelBase : public CImageFilterKernel<CRTP,CFloatingPointOnlyImageFilterKernelBase::value_type>, public CFloatingPointOnlyImageFilterKernelBase, public CSeparableImageFilterKernelBase, CIsotropicImageFilterKernelBase<Support>
{
		using StaticPolymorphicBase = CImageFilterKernel<CRTP,value_type>;

	public:
		CFloatingPointIsotropicSeparableImageFilterKernelBase() : StaticPolymorphicBase(symmetric_support,symmetric_support) {}
		

		template<class PreFilter, class PostFilter>
		struct sample_functor_t
		{
				sample_functor_t(const CRTP* __this, PreFilter& _preFilter, PostFilter& _postFilter) :
					_this(__this), preFilter(_preFilter), postFilter(_postFilter) {}

				void operator()(value_type* windowSample, core::vectorSIMDf& relativePosAndFactor, const core::vectorSIMDi32& globalTexelCoord);

			private:
				const CRTP* _this;
				PreFilter& preFilter;
				PostFilter& postFilter;
		};

		// this is the function that must be defined for each kernel
		template<class PreFilter, class PostFilter>
		inline auto create_sample_functor_t(PreFilter& preFilter, PostFilter& postFilter) const
		{
			return sample_functor_t(static_cast<const CRTP*>(this),preFilter,postFilter);
		}

	protected:
		// utility function so we dont evaluate `weight` function in children outside the support and just are able to return 0.f
		inline bool inDomain(float x) const
		{
			return core::abs(x)<isotropic_support;
		}
};

} // end namespace asset
} // end namespace irr

#endif