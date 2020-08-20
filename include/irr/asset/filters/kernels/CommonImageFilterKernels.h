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

// base class for all kernels that require pixels and arithmetic to be done in precise floats AND are separable AND have the same kernel function and support in each dimension AND have a rational support
// there's nothing special about having a rational support, we just use that type because there's no possibility of using `float` as a template parameter in C++
template<class CRTP,class Ratio=std::ratio<1,1> >
class CFloatingPointIsotropicSeparableImageFilterKernelBase : public CImageFilterKernel<CRTP,CFloatingPointOnlyImageFilterKernelBase::value_type>, public CFloatingPointOnlyImageFilterKernelBase
{
		using StaticPolymorphicBase = CImageFilterKernel<CRTP,value_type>;

	public:
		_IRR_STATIC_INLINE_CONSTEXPR bool is_separable = true;
		_IRR_STATIC_INLINE_CONSTEXPR float isotropic_support = float(Ratio::num)/float(Ratio::den);
		// utility constexpr array so we can pass a pointer to the base's constructor
		_IRR_STATIC_INLINE_CONSTEXPR float symmetric_support[3] = { isotropic_support,isotropic_support,isotropic_support };

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

		template<class PreFilter, class PostFilter>
		inline auto create_sample_functor_t(PreFilter& preFilter, PostFilter& postFilter) const
		{
			return sample_functor_t(reinterpret_cast<const CRTP*>(this),preFilter,postFilter);
		}

	protected:
		// utility function so we dont evaluate `weight` function in children outside the support and just are able to return 0.f
		inline bool inDomain(float x) const
		{
			return core::abs(x)<isotropic_support;
		}
};

// standard Box function, symmetric, value in the support is 1, integral is 1, so support must be [-1/2,1/2]
// to get box filters of different widths we can use it in composition inside `CScaledImageFilterKernel`
class CBoxImageFilterKernel : public CFloatingPointIsotropicSeparableImageFilterKernelBase<CBoxImageFilterKernel,std::ratio<1,2> >
{
		using Base = CFloatingPointIsotropicSeparableImageFilterKernelBase<CBoxImageFilterKernel,std::ratio<1,2> >;

	public:
		inline float weight(float x) const
		{
			return Base::inDomain(x) ? 1.f:0.f;
		}
};

// standard Triangle function, symmetric, peak in the support is 1 and at origin, integral is 1, so support must be [-1,1]
// to get box filters of different widths we can use it in composition inside `CScaledImageFilterKernel`
class CTriangleImageFilterKernel : public CFloatingPointIsotropicSeparableImageFilterKernelBase<CTriangleImageFilterKernel,std::ratio<1,1> >
{
		using Base = CFloatingPointIsotropicSeparableImageFilterKernelBase<CTriangleImageFilterKernel,std::ratio<1,1> >;

	public:
		inline float weight(float x) const
		{
			if (Base::inDomain(x))
				return 1.f-core::abs(x);
			return 0.f;
		}
};
/* Derivative at 0 not defined so we cannot use
class CDerivativeTriangleImageFilterKernel : public CFloatingPointIsotropicSeparableImageFilterKernelBase<CDerivativeTriangleImageFilterKernel, std::ratio<1, 1> >
{
	using Base = CFloatingPointIsotropicSeparableImageFilterKernelBase<CDerivativeTriangleImageFilterKernel, std::ratio<1, 1> >;

public:
	inline float weight(float x) const
	{
		if (Base::inDomain(x))
			return x<0.f ? 1.f:(x>0.f ? -1.f:0.f);
		return 0.f;
	}
};
*/

// Truncated Gaussian filter, with stddev = 1.0, if you want a different stddev then you need to scale it with `CScaledImageFilterKernel`
template<uint32_t support=3u>
class CGaussianImageFilterKernel : public CFloatingPointIsotropicSeparableImageFilterKernelBase<CGaussianImageFilterKernel<support>,std::ratio<support,1> >
{
		using Base = CFloatingPointIsotropicSeparableImageFilterKernelBase<CGaussianImageFilterKernel<support>,std::ratio<support,1> >;

	public:
		inline float weight(float x) const
		{
			if (Base::inDomain(x))
			{
				const float normalizationFactor = core::inversesqrt(2.f*core::PI<float>())/std::erff(core::sqrt<float>(2.f)*float(support));
				return normalizationFactor*exp2f(-0.72134752f*x*x);
			}
			return 0.f;
		}
};
template<uint32_t support=3u>
class CDerivativeGaussianImageFilterKernel : public CFloatingPointIsotropicSeparableImageFilterKernelBase<CDerivativeGaussianImageFilterKernel<support>,std::ratio<support,1> >
{
		using Base = CFloatingPointIsotropicSeparableImageFilterKernelBase<CDerivativeGaussianImageFilterKernel<support>,std::ratio<support,1> >;

	public:
		inline float weight(float x) const
		{
			return -x*CGaussianImageFilterKernel<support>::weight(x);
		}
};

// Kaiser filter, basically a windowed sinc, can be configured to different Kaiser window widths
template<uint32_t support=3u>
class CKaiserImageFilterKernel : public CFloatingPointIsotropicSeparableImageFilterKernelBase<CKaiserImageFilterKernel<support>,std::ratio<support,1> >
{
		using Base = CFloatingPointIsotropicSeparableImageFilterKernelBase<CKaiserImageFilterKernel<support>,std::ratio<support,1> >;

	public:
		// important constant, do not touch, do not tweak
		_IRR_STATIC_INLINE_CONSTEXPR float alpha = 3.f;

		inline float weight(float x) const
		{
			if (Base::inDomain(x))
			{
				const auto PI = core::PI<float>();
				return core::sinc(x*PI)*core::KaiserWindow(x,alpha,Base::isotropic_support);
			}
			return 0.f;
		}
};
template<uint32_t support=3u>
class CDerivativeKaiserImageFilterKernel : public CFloatingPointIsotropicSeparableImageFilterKernelBase<CDerivativeKaiserImageFilterKernel<support>,std::ratio<support,1> >
{
		using Base = CFloatingPointIsotropicSeparableImageFilterKernelBase<CDerivativeKaiserImageFilterKernel<support>,std::ratio<support,1> >;

	public:
		// important constant, do not touch, do not tweak
		_IRR_STATIC_INLINE_CONSTEXPR float alpha = 3.f;

		inline float weight(float x) const
		{
			if (Base::inDomain(x))
			{
				const auto PIx = core::PI<float>()*x;
				float f = core::sinc(PIx);
				float df = core::PI<float>()*core::d_sinc(PIx);
				float g = core::KaiserWindow(x, alpha, Base::isotropic_support);
				float dg = core::d_KaiserWindow(x, alpha, Base::isotropic_support);
				return df*g+f*dg;
			}
			return 0.f;
		}
};

// A standard Mitchell filter expressed as a convolution kernel, the standard has a support of [-2,2] the B and C template parameters are the same ones from the paper
template<class B=std::ratio<1,3>, class C=std::ratio<1,3> >
class CMitchellImageFilterKernel : public CFloatingPointIsotropicSeparableImageFilterKernelBase<CMitchellImageFilterKernel<B,C>,std::ratio<2,1> >
{
		using Base = CFloatingPointIsotropicSeparableImageFilterKernelBase<CMitchellImageFilterKernel<B,C>,std::ratio<2,1> >;

	public:
		inline float weight(float x) const
		{
			if (Base::inDomain(x))
			{
				x = core::abs(x);
				return core::mix(p0+x*x*(p2+x*p3),q0+x*(q1+x*(q2+x*q3)),x>=1.f);
			}
			return 0.f;
		}

	protected:
		_IRR_STATIC_INLINE_CONSTEXPR float b = float(B::num)/float(B::den);
		_IRR_STATIC_INLINE_CONSTEXPR float c = float(C::num)/float(C::den);
		_IRR_STATIC_INLINE_CONSTEXPR float p0 = (6.0f - 2.0f * b) / 6.0f;
		_IRR_STATIC_INLINE_CONSTEXPR float p2 = (-18.0f + 12.0f * b + 6.0f * c) / 6.0f;
		_IRR_STATIC_INLINE_CONSTEXPR float p3 = (12.0f - 9.0f * b - 6.0f * c) / 6.0f;
		_IRR_STATIC_INLINE_CONSTEXPR float q0 = (8.0f * b + 24.0f * c) / 6.0f;
		_IRR_STATIC_INLINE_CONSTEXPR float q1 = (-12.0f * b - 48.0f * c) / 6.0f;
		_IRR_STATIC_INLINE_CONSTEXPR float q2 = (6.0f * b + 30.0f * c) / 6.0f;
		_IRR_STATIC_INLINE_CONSTEXPR float q3 = (-b - 6.0f * c) / 6.0f;
};
template<class B=std::ratio<1,3>, class C=std::ratio<1,3> >
class CDerivativeMitchellImageFilterKernel : public CFloatingPointIsotropicSeparableImageFilterKernelBase<CDerivativeMitchellImageFilterKernel<B,C>,std::ratio<2,1> >
{
		using Base = CFloatingPointIsotropicSeparableImageFilterKernelBase<CDerivativeMitchellImageFilterKernel<B,C>,std::ratio<2,1> >;

	public:
		inline float weight(float x) const
		{
			if (Base::inDomain(x))
			{
				bool neg = x<0.f;
				x = core::abs(x);
				float retval = core::mix(x*(2.f*p2+3.f*x*p3),q1+x*(2.f*q2+3.f*x*q3),x>=1.f);
				return neg ? (-retval):retval;
			}
			return 0.f;
		}
};

} // end namespace asset
} // end namespace irr

#endif